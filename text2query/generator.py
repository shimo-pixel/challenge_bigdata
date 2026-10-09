from __future__ import annotations

import hashlib
import json
import os
import re
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from .io import dump_json, load_json

SYSTEM = """You compile natural language into precise read-only database queries. Return one JSON object only.
The competition's supplied schema, dictionary data, knowledge and training conventions are authoritative.
MYSQL and SQLite must qualify physical tables with database.table; PostgreSQL uses vessel_ais_data directly.
Database labels: MYSQL, POSTGRESQL, SQLite, Elasticsearch. Elasticsearch query must be a JSON-encoded string.
For a single source prediction use db,query. For sequential cross-source prediction use db1,query1,db2,query2;
query2 consumes query1's returned keys via IN (result). Never invent identifiers or copy example conditions blindly.
Use schema-aware joins, exact question filters, correct aggregate grouping, dates, units and sort/limit semantics.
Do not use the question ID for reasoning. Do not execute SQL, modify data, or use outside facts over provided data."""

class GenerationError(RuntimeError):
    pass


class FatalGenerationError(GenerationError):
    """Authentication, balance or configuration failures require outside action."""


def extract_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        for m in re.finditer(r'\{', text):
            try:
                obj, _ = decoder.raw_decode(text[m.start():])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None


class OpenAICompatibleGenerator:
    def __init__(self, cache_dir: Path | None = None):
        self.provider = os.environ.get('T2Q_PROVIDER', 'deepseek').lower()
        if self.provider not in {'deepseek', 'ollama'}:
            raise GenerationError('T2Q_PROVIDER must be deepseek or ollama')
        defaults = {'deepseek': ('https://api.deepseek.com', 'deepseek-v4-pro'),
                    'ollama': ('http://127.0.0.1:11434/v1', 'qwen3:8b')}
        default_base, default_model = defaults[self.provider]
        self.base = os.environ.get('T2Q_API_BASE', default_base).rstrip('/')
        self.model = os.environ.get('T2Q_MODEL', default_model)
        self.key = (os.environ.get('T2Q_API_KEY') or os.environ.get('DEEPSEEK_API_KEY', '')) if self.provider == 'deepseek' else ''
        self.cache_dir = cache_dir
        self.timeout = int(os.environ.get('T2Q_TIMEOUT', '300'))
        self.lock = threading.Lock()
        self.fatal_error = None
        self.stats = {'calls': 0, 'cache_hits': 0, 'retries': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

    @property
    def enabled(self) -> bool:
        return bool(self.base and (self.provider == 'ollama' or self.key))

    def _bump(self, **values):
        with self.lock:
            for key, value in values.items():
                self.stats[key] = self.stats.get(key, 0) + value

    def _event(self, event: dict):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self.lock, (self.cache_dir/'events.jsonl').open('a') as handle:
                handle.write(json.dumps({'time':time.time(),**event})+'\n')

    def _stream_response(self, response, started: float) -> dict:
        content, model, usage, finish = [], self.model, {}, None
        for raw in response:
            if time.monotonic()-started > self.timeout:
                raise TimeoutError('generation deadline exceeded')
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            message=line[5:].strip()
            if message=='[DONE]':
                break
            chunk=json.loads(message)
            model=chunk.get('model',model)
            if chunk.get('usage'): usage=chunk['usage']
            for choice in chunk.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content'): content.append(delta['content'])
                if choice.get('finish_reason'): finish=choice['finish_reason']
        if finish is None:
            raise GenerationError('incomplete stream')
        return {'model':model,'usage':usage,'choices':[{'finish_reason':finish,'message':{'content':''.join(content)}}]}

    def generate(self, prompt: str, temperature: float = 0.0, effort: str = 'high') -> dict:
        if not self.enabled:
            raise GenerationError('Set T2Q_API_KEY or DEEPSEEK_API_KEY, or use T2Q_PROVIDER=ollama for a local model')
        body = {'model': self.model, 'reasoning_effort': effort,
                'max_tokens': int(os.environ.get('T2Q_MAX_TOKENS', '16384')),
                'response_format': {'type': 'json_object'},
                'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]}
        if self.provider == 'deepseek' and 'deepseek' in self.model:
            body['thinking'] = {'type': 'enabled'}
        cache_key = hashlib.sha256(json.dumps({'endpoint': self.base, 'body': body}, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f'{cache_key}.json' if self.cache_dir else None
        if path and path.exists():
            self._bump(cache_hits=1)
            return load_json(path)['prediction']
        if self.fatal_error:
            raise FatalGenerationError(self.fatal_error)
        last_error = ''
        retry_after = 0.0
        for attempt in range(4):
            if attempt:
                self._bump(retries=1)
                delay=max(retry_after,min(float(os.environ.get('T2Q_RETRY_BASE','5'))*2**attempt,60))
                time.sleep(delay+random.uniform(0,2))
            transport={**body,'stream':True,'stream_options':{'include_usage':True}}
            encoded=json.dumps(transport,ensure_ascii=False).encode()
            headers = {'Content-Type': 'application/json'}
            if self.key:
                headers['Authorization'] = f'Bearer {self.key}'
            request = urllib.request.Request(f'{self.base}/chat/completions', data=encoded, headers=headers)
            try:
                self._bump(calls=1)
                started=time.monotonic()
                self._event({'event':'start','key':cache_key,'attempt':attempt+1,'effort':body['reasoning_effort']})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = self._stream_response(response,started)
                usage = payload.get('usage', {})
                self._event({'event':'complete','key':cache_key,'elapsed_seconds':round(time.monotonic()-started,2),'usage':usage})
                self._bump(prompt_tokens=usage.get('prompt_tokens', 0), completion_tokens=usage.get('completion_tokens', 0))
                choice = payload['choices'][0]
                if choice.get('finish_reason') == 'length':
                    body['max_tokens'] = min(body['max_tokens'] * 2, 65536)
                    if body['reasoning_effort']=='max': body['reasoning_effort']='high'
                    last_error = 'truncated generation'
                    continue
                result = extract_object(choice['message'].get('content') or '')
                if result is None:
                    last_error = 'model returned no JSON object'
                    continue
                if path:
                    dump_json(path, {'prediction': result, 'model': payload.get('model', self.model), 'usage': usage,
                                     'finish_reason': choice.get('finish_reason'), 'actual_effort':body['reasoning_effort'],
                                     'actual_max_tokens':body['max_tokens']})
                return result
            except urllib.error.HTTPError as exc:
                last_error = f'HTTP {exc.code}'
                try:
                    info=json.loads(exc.read().decode('utf-8'))
                    message=str(info.get('error',info))[:1000]
                except (ValueError,OSError):
                    message=''
                if self.key:
                    message = message.replace(self.key, '[REDACTED]')
                message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',message)[:200]
                self._event({'event':'retry','key':cache_key,'error':last_error,'message':message,'attempt':attempt+1})
                try: retry_after=min(120,float(exc.headers.get('Retry-After','0')))
                except (ValueError,TypeError): retry_after=0
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    with self.lock: self.fatal_error=last_error
                    raise FatalGenerationError(last_error) from None
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, json.JSONDecodeError, GenerationError) as exc:
                last_error = type(exc).__name__
                self._event({'event':'retry','key':cache_key,'error':last_error,'attempt':attempt+1})
                body['reasoning_effort']={'max':'high','high':'low'}.get(body['reasoning_effort'],body['reasoning_effort'])
        raise GenerationError(f'Generation exhausted retries: {last_error}')
