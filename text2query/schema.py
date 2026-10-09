from __future__ import annotations

import json
import re
import zipfile
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from .io import dump_json

IDENT = r'(?:`[^`]+`|"[^"]+"|[\w-]+)'
CREATE_HEAD_RE = re.compile(rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>{IDENT}(?:\.{IDENT})?)\s*(?P<open>\()', re.I)

@dataclass
class Table:
    source: str
    database: str
    name: str
    columns: list[str]
    ddl: str

    @property
    def key(self) -> str:
        return f'{self.database}.{self.name}' if self.database else self.name


def split_sql(text: str, delimiter: str = ',') -> list[str]:
    """Split at top level without breaking quoted literals or expressions."""
    pieces, start, depth, quote, i = [], 0, 0, '', 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == '\\':
                i += 2
                continue
            if c == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = ''
        elif c in "'\"`":
            quote = c
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == delimiter and depth == 0:
            pieces.append(text[start:i].strip())
            start = i + 1
        i += 1
    pieces.append(text[start:].strip())
    return pieces


def _columns(body: str) -> list[str]:
    body = re.sub(r'--[^\n]*', '', body)
    result = []
    for field in split_sql(body):
        if re.match(r'(?:PRIMARY|FOREIGN|UNIQUE|KEY|CONSTRAINT|CHECK|INDEX)\b', field, re.I):
            continue
        m = re.match(rf'({IDENT})\s+', field)
        if m:
            result.append(m.group(1).strip('`"'))
    return result


def parse_sql(text: str, source: str, default_database: str = '') -> list[Table]:
    positions = [(m.start(), m.group(1)) for m in re.finditer(r'(?:CREATE\s+DATABASE\s+(?:IF\s+NOT\s+EXISTS\s+)?|数据库:\s*)([\w-]+)', text, re.I)]
    tables = []
    for m in CREATE_HEAD_RE.finditer(text):
        database = default_database
        for pos, db in positions:
            if pos > m.start():
                break
            database = db
        depth, quote, end, i = 0, '', None, m.start('open')
        while i < len(text):
            c = text[i]
            if quote:
                if c == '\\':
                    i += 2
                    continue
                if c == quote:
                    if i + 1 < len(text) and text[i + 1] == quote:
                        i += 2
                        continue
                    quote = ''
            elif c in "'\"`":
                quote = c
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end is None:
            raise ValueError(f'Unbalanced DDL: {m.group("name")}')
        names = [x.strip('`"') for x in re.findall(IDENT, m.group('name'))]
        if len(names) == 2:
            database, name = names
        else:
            name = names[0]
        tail = re.match(r'\s*ENGINE[^;]*;', text[end + 1:], re.I)
        stop = end + 1 + (tail.end() if tail else 0)
        tables.append(Table(source, database, name, _columns(text[m.end():end]), text[m.start():stop]))
    return tables


def dictionary_statements(text: str) -> list[dict]:
    result = []
    for statement in split_sql(text, ';'):
        m = re.search(rf'INSERT\s+INTO\s+({IDENT}(?:\.{IDENT})?)\s*\(', statement, re.I)
        if m:
            result.append({'table': m.group(1).replace('`', '').replace('"', '').split('.')[-1], 'sql': statement[m.start():] + ';'})
    return result


def build_catalog(archive: Path, administrative: Path | None = None) -> dict:
    with zipfile.ZipFile(archive) as z:
        core = z.read('Schema/core_schema.txt').decode('gb18030')
        common = z.read('Schema/commom_schema.txt').decode('utf-8-sig')
        bird = z.read('Schema/bird_schemas.txt').decode('utf-8-sig')
        knowledge = json.loads(z.read('knowledge.json'))
    tables = parse_sql(core, 'MYSQL') + parse_sql(common, 'MYSQL', 'common') + parse_sql(bird, 'SQLite')
    for t in tables:
        if t.name == 'vessel_ais_data':
            t.source, t.database = 'POSTGRESQL', ''
    es = re.search(r'PUT\s+/([\w-]+)(.*?)(?=CREATE\s+TABLE|\Z)', core, re.S | re.I)
    if es:
        fields = re.findall(r'"([A-Za-z_][\w]*)"\s*:\s*\{\s*"type"', es.group(2))
        tables.append(Table('Elasticsearch', '', es.group(1), fields, es.group(0)))
    dictionaries = dictionary_statements(common)
    if administrative:
        if administrative.suffix.lower() == '.rar':
            import rarfile
            with rarfile.RarFile(administrative) as z:
                names = [n for n in z.namelist() if n.lower().endswith('.sql')]
                if not names:
                    raise ValueError('Administrative archive has no SQL')
                try:
                    raw = b'\n'.join(z.read(n) for n in names)
                except rarfile.RarCannotExec:
                    tool = shutil.which('unrar-free')
                    if not tool:
                        raise ValueError('Install unrar/bsdtar/unrar-free, or pass the extracted administrative SQL') from None
                    with tempfile.TemporaryDirectory(prefix='t2q-admin-') as directory:
                        subprocess.run([tool, '-x', str(administrative.resolve()), directory], check=True, capture_output=True)
                        files = sorted(Path(directory).rglob('*.sql'))
                        if not files:
                            raise ValueError('Archive extraction returned no SQL')
                        raw = b'\n'.join(p.read_bytes() for p in files)
        else:
            raw = administrative.read_bytes()
        try:
            admin = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            admin = raw.decode('gb18030')
        dictionaries = [d for d in dictionaries if d['table'] != 'administrative_division']
        dictionaries.extend(dictionary_statements(admin))
    return {'version': 2, 'tables': [asdict(t) for t in tables], 'knowledge': knowledge,
            'dictionaries': dictionaries, 'administrative_data_loaded': administrative is not None}


def write_catalog(archive: Path, destination: Path, administrative: Path | None = None) -> None:
    dump_json(destination, build_catalog(archive, administrative))
