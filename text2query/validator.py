from __future__ import annotations

import json
import re
from dataclasses import dataclass
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

DB_LABELS = {'MYSQL', 'POSTGRESQL', 'SQLite', 'Elasticsearch'}
DIALECTS = {'MYSQL': 'mysql', 'POSTGRESQL': 'postgres', 'SQLite': 'sqlite'}

@dataclass
class Validation:
    ok: bool
    errors: list[str]


def clean_query(value: str) -> str:
    return re.sub(r'^```(?:sql|json)?\s*|\s*```$', '', value.strip(), flags=re.I).strip()


def _es_validate(query: str, catalog: dict) -> list[str]:
    try:
        data = json.loads(query)
    except (json.JSONDecodeError, TypeError):
        return ['invalid Elasticsearch JSON']
    if not isinstance(data, dict) or not data:
        return ['Elasticsearch query must be a nonempty object']
    fields = {c for t in catalog['tables'] if t['source'] == 'Elasticsearch' for c in t['columns']}
    errors = []
    roots = {'query','aggs','aggregations','_source','sort','size','from','fields','script_fields',
             'runtime_mappings','post_filter','track_total_hits','min_score','search_after','timeout',
             'highlight','collapse','explain','stored_fields'}
    for key in data:
        if key not in roots:
            errors.append(f'unsupported Elasticsearch root key: {key}')
    for key in ('size','from'):
        if key in data and (not isinstance(data[key],int) or isinstance(data[key],bool) or data[key]<0):
            errors.append(f'Elasticsearch {key} must be a nonnegative integer')
    def check(field):
        if not isinstance(field, str):
            errors.append('Elasticsearch field must be a string')
        elif field not in fields and not (field == 'name.keyword') and field not in {'_score', '_id', '_doc', '*'}:
            errors.append(f'unknown Elasticsearch field: {field}')
    numeric = {'bed_count','doctor_count','nurse_count','ambulance_count','department_count', 'annual_outpatient','rating','founded_year'}
    booleans = {'has_icu','is_teaching_hospital','emergency_24h'}
    def walk(node):
        if isinstance(node, list):
            for x in node: walk(x)
        elif isinstance(node, dict):
            for op, value in node.items():
                if op == 'field': check(value)
                if op in {'term', 'terms', 'match', 'match_phrase', 'range', 'prefix', 'wildcard', 'regexp', 'fuzzy'} and isinstance(value, dict):
                    for f, val in value.items():
                        check(f)
                        scalar = val.get('value') if isinstance(val, dict) and 'value' in val else val
                        if f in booleans and op == 'term' and not isinstance(scalar, bool):
                            errors.append(f'{f} requires a boolean')
                        if f in numeric and op == 'range' and isinstance(val, dict):
                            for bound, v in val.items():
                                if bound in {'gt','gte','lt','lte'} and (not isinstance(v, (int,float)) or isinstance(v,bool)):
                                    errors.append(f'{f} requires numeric bounds')
                if op in {'geo_distance', 'geo_bounding_box', 'geo_polygon'} and isinstance(value, dict):
                    for f in value:
                        if f not in {'distance','distance_type','validation_method','ignore_unmapped','boost'}: check(f)
                if op == 'exists' and isinstance(value,dict): check(value.get('field'))
                if op == '_source':
                    if isinstance(value,str): check(value)
                    elif isinstance(value,list):
                        for f in value: check(f)
                    elif isinstance(value,dict):
                        for group in ('includes','excludes'):
                            for f in value.get(group,[]): check(f)
                if op == 'sort':
                    for entry in (value if isinstance(value,list) else [value]):
                        if isinstance(entry,str): check(entry)
                        elif isinstance(entry,dict):
                            for f in entry:
                                if f != '_geo_distance': check(f)
                walk(value)
    walk(data)
    return list(dict.fromkeys(errors))


def validate_query(db: str, query: str, catalog: dict, second_stage: bool = False) -> Validation:
    if db not in DB_LABELS:
        return Validation(False, [f'unsupported db label: {db}'])
    if not isinstance(query,str) or not query.strip():
        return Validation(False, ['query must be a nonempty string'])
    query = clean_query(query)
    if db == 'Elasticsearch':
        errors = _es_validate(query, catalog)
        if second_stage and 'result' not in query:
            errors.append('second-stage cross-source query must consume result')
        return Validation(not errors, errors)
    errors = []
    try:
        statements = sqlglot.parse(query, read=DIALECTS[db], error_level='RAISE')
        if len(statements) != 1 or statements[0] is None:
            return Validation(False, ['exactly one SQL query required'])
        tree = statements[0]
    except sqlglot.errors.SqlglotError as e:
        return Validation(False, [f'SQL parse error: {str(e)[:300]}'])
    if not isinstance(tree, exp.Query) or tree.find(exp.Into) or tree.find(exp.Lock):
        errors.append('only read-only SELECT/WITH queries allowed')
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)):
            errors.append('mutating SQL is forbidden')
    if second_stage:
        refs = [x for x in tree.find_all(exp.Column) if x.name.lower() == 'result']
        if not any(x.find_ancestor(exp.In) for x in refs):
            errors.append('second-stage cross-source query must contain IN (result)')
    tables = {(t['database'].lower(), t['name'].lower()): t for t in catalog['tables'] if t['source'] == db}
    try:
        scopes = list(traverse_scope(tree))
    except sqlglot.errors.SqlglotError:
        scopes = []
    for scope in scopes:
        source_columns = {}
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                key = (source.db.lower(), source.name.lower())
                if db in {'MYSQL','SQLite'} and not source.db:
                    errors.append(f'physical table must be database-qualified: {source.name}')
                table = tables.get(key)
                if db == 'POSTGRESQL':
                    table = next((t for (_,n),t in tables.items() if n == source.name.lower()),None)
                if table is None:
                    errors.append(f'unknown {db} table: {source.sql()}')
                else:
                    # Empty columns mean unavailable metadata (e.g. small test catalogs).
                    source_columns[alias.lower()] = {x.lower() for x in table.get('columns', [])} or None
            elif isinstance(source, Scope):
                names = source.expression.named_selects
                source_columns[alias.lower()] = None if '*' in names else {x.lower() for x in names}
        projection_aliases = {x.alias.lower() for x in scope.expression.expressions if x.alias}
        for column in scope.columns:
            # Scope.columns also includes unqualified nested external columns;
            # validate each column in its own SELECT, not again in the parent.
            if column.find_ancestor(exp.Query) is not scope.expression:
                continue
            name, alias = column.name.lower(), column.table.lower()
            if name == '*' or name == 'result' and second_stage and not alias:
                continue
            if alias and alias in source_columns:
                available = source_columns[alias]
                if available is not None and name not in available:
                    errors.append(f'unknown column: {column.sql()}')
            elif alias:
                parent = scope.parent
                available = False
                while parent:
                    if alias in {key.lower() for key in parent.sources}:
                        available = True
                    parent = parent.parent
                if not available:
                    errors.append(f'unknown table alias: {alias}')
            elif not alias and name not in projection_aliases:
                known_sets = list(source_columns.values())
                if known_sets and all(x is not None for x in known_sets) and not any(name in x for x in known_sets):
                    # Correlated subqueries may refer to their parent's columns.
                    parent = scope.parent
                    correlated = False
                    while parent:
                        for source in parent.sources.values():
                            if isinstance(source,exp.Table):
                                t = tables.get((source.db.lower(),source.name.lower()))
                                if t and name in {x.lower() for x in t.get('columns',[])}: correlated = True
                        parent = parent.parent
                    if not correlated:
                        errors.append(f'unknown column: {column.sql()}')
    errors = list(dict.fromkeys(errors))
    return Validation(not errors, errors)


def validate_record(record: dict, catalog: dict) -> Validation:
    if not isinstance(record,dict):
        return Validation(False,['record must be an object'])
    single = {'question_id','db','query'}
    cross = {'question_id','db1','query1','db2','query2'}
    keys = set(record)
    if keys == single:
        errors = validate_query(record['db'],record['query'],catalog).errors
    elif keys == cross:
        errors = validate_query(record['db1'],record['query1'],catalog).errors + validate_query(record['db2'],record['query2'],catalog,True).errors
        if record['db1'] == record['db2']:
            errors.append('same engine queries should use a single query')
    else:
        return Validation(False,['unexpected submission keys; use question_id,db,query or question_id,db1,query1,db2,query2'])
    if not isinstance(record.get('question_id'),int) or isinstance(record.get('question_id'),bool):
        errors.append('question_id must be an integer')
    return Validation(not errors,errors)
