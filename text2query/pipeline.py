from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from .generator import FatalGenerationError, GenerationError, OpenAICompatibleGenerator
from .io import dump_json, load_json
from .retrieval import CharTfidf
from .schema import write_catalog, split_sql
from .validator import clean_query, validate_record, DIALECTS

PIPELINE_VERSION = 'v2.6'


def queries(row: dict) -> list[str]:
    return [row[k] for k in ('query','query1','query2') if isinstance(row.get(k),str)]


def referenced_databases(row: dict) -> set[str]:
    result = set()
    for key, dbkey in [('query','db'),('query1','db1'),('query2','db2')]:
        if row.get(dbkey) not in DIALECTS:
            continue
        try:
            tree = sqlglot.parse_one(row[key],read=DIALECTS[row[dbkey]])
            result.update(t.db.lower() for t in tree.find_all(exp.Table) if t.db and t.db.lower() != 'common')
        except (sqlglot.errors.SqlglotError,KeyError):
            result.update(re.findall(r'\b(?:FROM|JOIN)\s+([\w]+)\.',row.get(key,''),re.I))
    return result


def question_key(row: dict) -> str:
    return re.sub(r'[\s，。！？,.!?]', '', row['question']).lower()


def split_training(train: list[dict], seed: int = 42, fraction: float = .1):
    grouped = defaultdict(list)
    for row in train:
        grouped[question_key(row)].append(row)
    strata = defaultdict(list)
    for group in grouped.values():
        row = group[0]
        label = (row.get('db','CROSS'), tuple(sorted(referenced_databases(row))))
        strata[label].append(group)
    rng = random.Random(seed)
    fit, heldout = [], []
    for label in sorted(strata, key=str):
        groups = sorted(strata[label],key=lambda g:g[0]['question_id'])
        rng.shuffle(groups)
        n = max(1,round(len(groups)*fraction)) if len(groups)>1 else 0
        heldout.extend(r for group in groups[:n] for r in group)
        fit.extend(r for group in groups[n:] for r in group)
    return sorted(fit,key=lambda r:r['question_id']),sorted(heldout,key=lambda r:r['question_id'])


def stratified_sample(rows: list[dict], limit: int | None, seed: int = 42) -> list[dict]:
    if not limit or limit >= len(rows): return rows
    strata = defaultdict(list)
    for row in rows:
        strata[(row.get('db','CROSS'),tuple(sorted(referenced_databases(row))))].append(row)
    rng = random.Random(seed)
    groups = list(strata.values())
    for group in groups: rng.shuffle(group)
    # Round-robin covers uncommon domains before proportional remainder.
    sampled = []
    while groups and len(sampled)<limit:
        groups.sort(key=lambda x:len(x),reverse=True)
        for group in groups:
            if group and len(sampled)<limit: sampled.append(group.pop())
        groups = [g for g in groups if g]
    return sorted(sampled,key=lambda r:r['question_id'])


def normalize_query(db: str, query: str) -> str:
    if db == 'Elasticsearch':
        try: return json.dumps(json.loads(query),sort_keys=True,separators=(',',':'),ensure_ascii=False)
        except (ValueError,TypeError): return query.strip()
    try:
        return sqlglot.parse_one(query,read=DIALECTS[db]).sql(dialect=DIALECTS[db],normalize=True,pretty=False)
    except (sqlglot.errors.SqlglotError,KeyError):
        return re.sub(r'\s+',' ',query.strip())


def same_prediction(pred: dict, actual: dict) -> bool:
    if pred.get('db','CROSS') != actual.get('db','CROSS'): return False
    if 'db' in actual:
        return normalize_query(actual['db'],pred.get('query','')) == normalize_query(actual['db'],actual['query'])
    return all(pred.get('db'+str(i))==actual['db'+str(i)] and
               normalize_query(actual['db'+str(i)],pred.get('query'+str(i),''))==normalize_query(actual['db'+str(i)],actual['query'+str(i)]) for i in (1,2))


class Pipeline:
    def __init__(self, workdir: Path, train: list[dict] | None = None):
        self.workdir = workdir
        self.catalog = load_json(workdir/'catalog.json')
        self.train = load_json(workdir/'QA_train.json') if train is None else train
        self.test = load_json(workdir/'QA_test.json')
        self.retriever = CharTfidf(r['question'] for r in self.train)
        self.row_databases = [referenced_databases(r) for r in self.train]
        self.by_db = defaultdict(set)
        self.by_database = defaultdict(set)
        for i,row in enumerate(self.train):
            self.by_db[row.get('db','CROSS')].add(i)
            for db in self.row_databases[i]: self.by_database[db].add(i)
        self.tables = self.catalog['tables']
        self.table_retriever = CharTfidf(f"{t['database']} {t['name']} {' '.join(t['columns'])} {t['ddl']}" for t in self.tables)
        self.knowledge = self.catalog['knowledge']
        self.knowledge_retriever = CharTfidf(k['knowledge_text'] for k in self.knowledge)
        self.value_evidence = self._value_evidence()
        self.generator = OpenAICompatibleGenerator(workdir/'cache')
        self.directory = self._directory()
        self.fingerprint = hashlib.sha256(json.dumps({'version': PIPELINE_VERSION, 'catalog': self.catalog,
                    'train': self.train, 'model':self.generator.model,'base':self.generator.base},sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]

    @classmethod
    def prepare(cls, archive: Path, workdir: Path, administrative: Path | None = None):
        workdir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in ('QA_train.json','QA_test.json','knowledge.json'):
                (workdir/name).write_bytes(z.read(name))
        write_catalog(archive,workdir/'catalog.json',administrative)

    def _directory(self) -> str:
        groups = defaultdict(list)
        for t in self.tables: groups[(t['source'],t['database'])].append(t['name'])
        lines=[]
        for (source,db),names in groups.items():
            samples = [self.train[i]['question'] for i in sorted(self.by_database[db])[:2]] if db else []
            lines.append(f"{source} {db or '(no database prefix)'}: {', '.join(names)}" + ('; example topics: '+' / '.join(samples) if samples else ''))
        return '\n'.join(lines)

    def _value_evidence(self) -> dict:
        """Values seen in fit queries only; these are evidence, not exhaustive enums."""
        evidence = defaultdict(lambda: defaultdict(Counter))
        metadata = {(t['source'], t['database'].lower(), t['name'].lower()): t for t in self.tables}
        for row in self.train:
            for key, label in [('query','db'),('query1','db1'),('query2','db2')]:
                source = row.get(label)
                if source not in DIALECTS:
                    continue
                try:
                    tree = sqlglot.parse_one(row[key], read=DIALECTS[source])
                    for scope in traverse_scope(tree):
                        physical = {a.lower(): t for a,t in scope.sources.items() if isinstance(t,exp.Table)}
                        for condition in scope.expression.find_all(exp.EQ, exp.In):
                            if condition.find_ancestor(exp.Query) is not scope.expression:
                                continue
                            column = condition.this
                            if not isinstance(column,exp.Column):
                                continue
                            if column.table:
                                table = physical.get(column.table.lower())
                            else:
                                candidates = [t for t in physical.values() if column.name.lower() in
                                              {c.lower() for c in metadata.get((source,t.db.lower(),t.name.lower()),{}).get('columns',[])}]
                                table = candidates[0] if len(candidates)==1 else None
                            if table is None:
                                continue
                            literals = [condition.expression] if isinstance(condition,exp.EQ) else condition.expressions
                            for literal in literals:
                                if isinstance(literal,exp.Literal) and literal.is_string:
                                    evidence[(source,table.db.lower(),table.name.lower())][column.name.lower()][literal.this] += 1
                except (sqlglot.errors.SqlglotError, KeyError):
                    continue
        return evidence

    def route(self, question: str, question_id: int = 0, exclude_id: int | None = None) -> str:
        allowed = {i for i,r in enumerate(self.train) if r['question_id']!=exclude_id} if exclude_id is not None else None
        hits=self.retriever.search(question,12,allowed)
        votes=Counter()
        for h in hits: votes[self.train[h.index].get('db','CROSS')]+=h.score
        return votes.most_common(1)[0][0] if votes else 'SQLite'

    def _hints(self, question: str, question_id: int | None = None) -> dict:
        hits=self.retriever.search(question,20)
        source_votes, db_votes=Counter(),Counter()
        for h in hits:
            source_votes[self.train[h.index].get('db','CROSS')]+=h.score**2
            for db in self.row_databases[h.index]: db_votes[db]+=h.score**2
        route=source_votes.most_common(1)[0][0] if source_votes else 'SQLite'
        cross_example=next((self.train[h.index] for h in hits if self.train[h.index].get('db','CROSS')=='CROSS'),None)
        sources=[route] if route!='CROSS' else list(dict.fromkeys(cross_example[k] for k in ('db1','db2')))
        databases=[x for x,_ in db_votes.most_common(2)]
        if len(databases)>1 and db_votes[databases[1]] < .3*db_votes[databases[0]]: databases=databases[:1]
        hint={'sources':sources,'databases':databases,'route':route,'complex':route=='CROSS'}
        # Many education questions omit the year entirely. Train metadata can
        # disambiguate their dataset batch; never let this choose the engine or
        # override an explicit year / a request spanning years.
        education={'education_2024','education_2025'}
        if question_id is not None and education.intersection(databases) and not re.search(r'20\d{2}|跨年|两年|各年|历年|所有年|不同年|上一年|前一年|同比|两个年度|每年|每个年|按年|年度',question):
            nearby=Counter()
            candidates=sorted((abs(row['question_id']-question_id),row['question_id'],next(iter(dbs)))
                              for row,dbs in zip(self.train,self.row_databases)
                              if row['question_id']!=question_id and abs(row['question_id']-question_id)<=30 and len(dbs)==1 and dbs<=education)
            for _,_,database in candidates[:8]: nearby[database]+=1
            if sum(nearby.values())>=5:
                default,count=nearby.most_common(1)[0]
                if count/sum(nearby.values())>=.85:
                    hint['default_database']=default
                    hint['default_note']='Weak dataset-batch prior ONLY for an unspecified education year; explicit year and cross-year requests take precedence.'
                    hint['databases']=[default]
        return hint

    def _examples(self, question: str, route: str | None = None, exclude_id: int | None = None, plan: dict | None = None) -> list[dict]:
        allowed=set(range(len(self.train)))
        if route: allowed &= self.by_db[route]
        if plan and plan.get('databases'):
            domain=set().union(*(self.by_database[d] for d in plan['databases']))
            if domain: allowed &= domain
        if exclude_id is not None: allowed={i for i in allowed if self.train[i]['question_id']!=exclude_id}
        hits=self.retriever.search(question,8,allowed)
        # Avoid spending the example budget on duplicate questions.
        result,seen=[],set()
        for h in hits:
            row=self.train[h.index]
            if question_key(row) not in seen:
                result.append({k:v for k,v in row.items() if k!='question_id'})
                seen.add(question_key(row))
            if len(result)==6: break
        return result

    def _selected_tables(self,question: str,plan: dict,examples: list[dict]) -> list[dict]:
        sources=set(plan.get('sources',[]))
        databases=set(plan.get('databases',[]))
        keys=set(plan.get('tables',[]))
        chosen=set()
        # Providing a full selected database avoids losing English-only BIRD tables.
        for i,t in enumerate(self.tables):
            full=f"{t['database']}.{t['name']}" if t['database'] else t['name']
            if t['source'] in sources and (t['database'] in databases or not t['database']): chosen.add(i)
            if full in keys: chosen.add(i)
        example_sql=' '.join(q for e in examples for q in queries(e))
        for h in self.table_retriever.search(question,8):
            table=self.tables[h.index]
            if table['source'] in sources and (not databases or table['database'] in databases or table['database']=='common'):
                chosen.add(h.index)
        for i,t in enumerate(self.tables):
            full=f"{t['database']}.{t['name']}" if t['database'] else t['name']
            if t['source'] in sources and (full in example_sql or t['database'] in databases and re.search(r'\b'+re.escape(t['name'])+r'\b',example_sql)): chosen.add(i)
        if 'POSTGRESQL' in sources:
            # AIS lacks ownership/company attributes. Expose the registry bridge
            # even if initial retrieval guessed a single PostgreSQL source.
            chosen.update(i for i,t in enumerate(self.tables) if t['source']=='MYSQL' and t['database']=='transportation' and t['name'] in {'vessel','shipping_company'})
        # Follow schema FK references once, including common dictionary joins.
        ddl='\n'.join(self.tables[i]['ddl'] for i in chosen)
        refs=re.findall(r'REFERENCES\s+([\w.`"-]+)',ddl,re.I)
        for i,t in enumerate(self.tables):
            full=f"{t['database']}.{t['name']}"
            if any(ref.replace('`','').replace('"','') in {full,t['name']} for ref in refs) and t['source'] in sources: chosen.add(i)
        return [self.tables[i] for i in sorted(chosen)]

    def _dictionary_context(self,question: str,tables: list[dict]) -> str:
        names={t['name'] for t in tables}
        parts=[]
        for d in self.catalog.get('dictionaries',[]):
            if d['table']!='administrative_division':
                if d['table'] in names: parts.append(d['sql'])
                continue
            # Pick exact place-name matches, not model geographic knowledge.
            statement=d['sql']; head,sep,values=statement.partition('VALUES')
            if not sep: head,sep,values=statement.partition('values')
            if not sep: continue
            selected=[]
            for row in split_sql(values.rstrip(';')):
                labels=re.findall(r"'([^']*)'",row)
                matches=[x for x in labels[1:6] if len(x)>=2 and (x in question or x.rstrip('省市区县') in question and len(x.rstrip('省市区县'))>=2)]
                if matches: selected.append(row)
            if selected: parts.append(head+'VALUES\n'+',\n'.join(selected[:100])+';')
        return '\n'.join(parts)

    def _context(self, plan: dict | str, question: str, examples: list[dict]) -> str:
        if isinstance(plan,str): plan={'sources':[plan] if plan!='CROSS' else ['MYSQL','POSTGRESQL'],'databases':list(set().union(*(referenced_databases(e) for e in examples)))}
        tables=self._selected_tables(question,plan,examples)
        ddls='\n\n'.join(f"[{t['source']}] TABLE {t['database']+'.' if t['database'] else ''}{t['name']}\n{t['ddl']}" for t in tables)
        main=set(plan.get('databases',[]))
        allowed={i for i,k in enumerate(self.knowledge) if not main or any(db in k['knowledge_text'] for db in main) or not re.search(r'数据库\s*[A-Za-z_]',k['knowledge_text'])}
        search=question+' '+' '.join(plan.get('tables',[]))
        hits=self.knowledge_retriever.search(search,12,allowed)
        global_hits=self.knowledge_retriever.search(question,6)
        indices=list(dict.fromkeys(h.index for h in hits+global_hits))
        knowledge='\n'.join(f"K{self.knowledge[i]['knowledge_id']}: {self.knowledge[i]['knowledge_text']}" for i in indices)
        value_lines=[]
        for table in tables:
            values=self.value_evidence.get((table['source'],table['database'].lower(),table['name'].lower()),{})
            for field,counter in values.items():
                frequent=[v for v,_ in counter.most_common(16)]
                mentioned=[v for v in counter if len(v)>=2 and v.lower() in question.lower()]
                selected=list(dict.fromkeys(mentioned+frequent))[:24]
                value_lines.append(f"{table['database']+'.' if table['database'] else ''}{table['name']}.{field}: {json.dumps(selected,ensure_ascii=False)}")
        value_text='\n'.join(value_lines)
        bridge=''
        if 'POSTGRESQL' in plan.get('sources',[]):
            bridge='\nCROSS-SOURCE LINK: MYSQL transportation.vessel.mmsi matches POSTGRESQL vessel_ais_data.mmsi. Vessel registry attributes and company data live in MYSQL. If these attributes are requested, revise a single-source hint to CROSS. Order stages so the final source can return the requested attributes: filter AIS first and return mmsi, then query MYSQL vessel/shipping_company with mmsi IN (result) when company output is needed. Pure AIS questions remain single-source.\n'
        return f"AUTHORITATIVE SCHEMA:\n{ddls}\n{bridge}\nAUTHORITATIVE DICTIONARY VALUES:\n{self._dictionary_context(question,tables)}\n\nVALUE EVIDENCE FROM TRAINING QUERIES (non-exhaustive; prefer observed spellings/codes when expressing the same meaning; never add an unrequested filter):\n{value_text}\n\nRETRIEVED KNOWLEDGE:\n{knowledge}"

    @staticmethod
    def _prediction(candidate: dict, question_id: int) -> dict:
        if isinstance(candidate.get('prediction'),dict): candidate=candidate['prediction']
        keys=('db','query') if 'db' in candidate else ('db1','query1','db2','query2')
        if not all(k in candidate for k in keys): raise GenerationError('Incomplete model prediction')
        output={k:candidate[k] for k in keys}
        output['question_id']=question_id
        for key in ('query','query1','query2'):
            if key in output:
                if isinstance(output[key],dict): output[key]=json.dumps(output[key],ensure_ascii=False,separators=(',',':'))
                if not isinstance(output[key],str): raise GenerationError('Query is not a string')
                output[key]=clean_query(output[key])
        return output

    @staticmethod
    def _offline_template(row: dict, examples: list[dict], route: str) -> dict:
        if not examples: raise GenerationError('No offline template available')
        keys=('db1','query1','db2','query2') if route=='CROSS' else ('db','query')
        return {**{k:examples[0][k] for k in keys},'question_id':row['question_id']}

    def _generate_prediction(self,prompt: str,question_id: int,effort: str='high') -> dict:
        for attempt in range(3):
            response=self.generator.generate(prompt,effort=effort)
            try:
                self._prediction(response,question_id)
                return response
            except GenerationError:
                prompt+='\nYour previous response omitted required prediction fields:\n'+json.dumps(response,ensure_ascii=False)
                prompt+='\nReturn a complete prediction object with db,query OR db1,query1,db2,query2. If routing missed an attribute, revise routing using the supplied cross-source links and schema. Do not return prediction:null. JSON only.'
        raise GenerationError('Model repeatedly returned incomplete prediction fields')

    def _generate_plan(self,prompt: str,question_id: int) -> dict:
        labels={'mysql','postgresql','sqlite','elasticsearch'}
        for attempt in range(3):
            response=self.generator.generate(prompt)
            plan=response['plan'] if isinstance(response.get('plan'),dict) else response
            if isinstance(plan.get('sources'),str): plan['sources']=[plan['sources']]
            sources=plan.get('sources')
            if isinstance(sources,list) and sources and all(isinstance(s,str) and s.strip().lower() in labels for s in sources):
                return plan
            try:
                early=self._prediction(response,question_id)
                sources=[early['db']] if 'db' in early else [early['db1'],early['db2']]
                if all(isinstance(s,str) and s.strip().lower() in labels for s in sources):
                    return {'sources':sources,'databases':sorted(referenced_databases(early)),
                            'tables':[], 'conditions':[], 'operations':[], 'complex':len(sources)>1,
                            'ambiguity':['Planner returned a query; routing extracted and generation repeated.']}
            except GenerationError:
                pass
            prompt+='\nYour previous response was not a complete route plan:\n'+json.dumps(response,ensure_ascii=False)
            prompt+='\nReturn a full JSON plan with sources (nonempty list of actual source labels), databases, tables, conditions, operations, complex, ambiguity. Choose sources from MYSQL, POSTGRESQL, SQLite, Elasticsearch. Do not return a partial SQL or DSL clause.'
        raise GenerationError('Model repeatedly returned an incomplete route plan')

    def predict_one(self,row: dict,offline: bool=False,review: bool=True) -> dict:
        if offline:
            route=self.route(row['question']);return self._offline_template(row,self._examples(row['question'],route),route)
        q=row['question'];hint=self._hints(q,row['question_id'])
        examples=self._examples(q,hint['route'],plan=hint)
        planning=f"QUESTION: {q}\nDATABASE DIRECTORY:\n{self.directory}\nRETRIEVAL HINT (may be wrong): {json.dumps(hint)}\n{self._context(hint,q,examples)}\nSIMILAR TRAINING EXAMPLES:\n{json.dumps(examples,ensure_ascii=False)}\n" + """First identify the exact requested output, sources, database years, tables, filters, joins, aggregation, date boundaries, units and ordering. This is a plan, not final SQL. Return JSON with sources (list of source labels), databases (list of actual database names), tables (list of database.table or unqualified PG/ES names), conditions (list), operations (list), complex (boolean), ambiguity (list). Same-engine joins are a single-source query. Distinguish administrative vessel/company data (MYSQL) from AIS observations (POSTGRESQL)."""
        plan=self._generate_plan(planning,row['question_id'])
        known_dbs={t['database'] for t in self.tables}
        labels={'mysql':'MYSQL','postgresql':'POSTGRESQL','sqlite':'SQLite','elasticsearch':'Elasticsearch'}
        plan['sources']=[labels.get(str(s).strip().lower(),str(s).strip()) for s in plan['sources']]
        plan['databases']=[d for d in plan.get('databases',[]) if d in known_dbs]
        plan['tables']=[x for x in plan.get('tables',[]) if isinstance(x,str)]
        if hint.get('default_database'):
            plan['default_database']=hint['default_database']
            plan['default_note']=hint['default_note']
            if plan['sources']==['MYSQL'] and plan['databases'] and set(plan['databases'])<={'education_2024','education_2025'}:
                # Bind the otherwise unresolved single-year default. Leave
                # source selection and multi-year plans to semantic routing.
                plan['databases']=[hint['default_database']]
                plan['tables']=list(dict.fromkeys(re.sub(r'education_202[45]\.',hint['default_database']+'.',t) for t in plan['tables']))
        route='CROSS' if len(set(plan['sources']))>1 else plan['sources'][0]
        examples=self._examples(q,route,plan=plan)
        context=self._context(plan,q,examples)
        prompt=f"QUESTION: {q}\nAPPROVED QUERY PLAN:\n{json.dumps(plan,ensure_ascii=False)}\n{context}\nSIMILAR TRAINING EXAMPLES (adapt their structure; replace conditions to answer this question):\n{json.dumps(examples,ensure_ascii=False)}\nGenerate the final query. Check every condition and requested output; include no extra unrequested filters. Return JSON {{\"prediction\":{{\"db\":\"...\",\"query\":\"...\"}}}} or the cross-source shape inside prediction."
        generated=self._generate_prediction(prompt,row['question_id'])
        candidate=self._prediction(generated,row['question_id'])
        validation=validate_record(candidate,self.catalog)
        traces={'plan':plan,'initial':candidate,'initial_errors':validation.errors}
        candidates=[candidate]
        complex_case=bool(plan.get('complex')) or route=='CROSS' or any(x in q for x in ('轨迹','跨年','上一年','同比','环比'))
        if complex_case and review:
            alternative=self._generate_prediction(prompt+'\nIndependently derive a second solution from the question and schema. Pay particular attention to join multiplicity, time ranges, geography and yearly routing.',row['question_id'],effort='max')
            candidates.append(self._prediction(alternative,row['question_id']))
        if review:
            checks=[{'prediction':c,'static_errors':validate_record(c,self.catalog).errors} for c in candidates]
            audit=f"QUESTION: {q}\n{context}\nREFERENCE TRAINING EXAMPLES:\n{json.dumps(examples,ensure_ascii=False)}\nCANDIDATE QUERIES:\n{json.dumps(checks,ensure_ascii=False)}\n"+"""Independently verify that each candidate answers this exact question. Check routing, every literal/filter, missing/extraneous filters, comparison direction, projection, grouping, DISTINCT/join multiplicity, date/year boundaries, coordinate order, meters/km, dictionary codes and cross-source output dependencies. Prefer competition training conventions when the question is underspecified. Static errors must be fixed. Return JSON {"needs_alternative": false, "issues": [brief concrete issues], "prediction": {final corrected submission fields}}. Set needs_alternative true only when materially uncertain about a query's semantics."""
            reviewed=self._generate_prediction(audit,row['question_id'])
            traces['review']=reviewed
            candidate=self._prediction(reviewed,row['question_id'])
            if reviewed.get('needs_alternative') and not complex_case:
                alternative=self._prediction(self._generate_prediction(prompt+'\nProduce an independent candidate resolving these semantic uncertainties: '+json.dumps(reviewed.get('issues',[]),ensure_ascii=False),row['question_id'],effort='max'),row['question_id'])
                final=self._generate_prediction(audit+'\nADDITIONAL INDEPENDENT CANDIDATE:\n'+json.dumps(alternative,ensure_ascii=False)+'\nResolve the uncertainties and return the final corrected prediction.',row['question_id'],effort='max')
                traces['selection']=final
                candidate=self._prediction(final,row['question_id'])
        for attempt in range(2):
            validation=validate_record(candidate,self.catalog)
            if validation.ok: break
            repair=self._generate_prediction(prompt+'\nRepair the following candidate without changing the requested semantics:\n'+json.dumps(candidate,ensure_ascii=False)+'\nSTATIC ERRORS:\n'+json.dumps(validation.errors,ensure_ascii=False),row['question_id'],effort='high')
            traces[f'repair{attempt}']=repair
            candidate=self._prediction(repair,row['question_id'])
        validation=validate_record(candidate,self.catalog)
        if not validation.ok: raise GenerationError('Unresolved query validation: '+'; '.join(validation.errors))
        traces['final']=candidate
        trace_path=self.workdir/'runs'/self.fingerprint/f"{hashlib.sha256(q.encode()).hexdigest()[:12]}-{'review' if review else 'single'}.json"
        dump_json(trace_path,traces)
        return candidate

    def predict(self,offline: bool=False,review: bool=True,workers: int=4,resume: bool=True,rows: list[dict] | None=None) -> list[dict]:
        rows=self.test if rows is None else rows
        config={'fingerprint':self.fingerprint,'offline':offline,'review':review,'rows':rows}
        run_id=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
        checkpoint=self.workdir/'runs'/run_id/'predictions.json'
        saved=load_json(checkpoint) if resume and checkpoint.exists() else {}
        complete={int(k):v for k,v in saved.items() if validate_record(v,self.catalog).ok}
        pending=[r for r in rows if r['question_id'] not in complete]
        errors=[]
        print(f'Run {run_id}: {len(complete)}/{len(rows)} resumed, {len(pending)} pending',flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(self.predict_one,row,offline,review):row for row in pending}
            for future in as_completed(futures):
                row=futures[future]
                try:
                    complete[row['question_id']]=future.result()
                    dump_json(checkpoint,{str(k):v for k,v in complete.items()})
                    print(f"completed {len(complete)}/{len(rows)} id={row['question_id']} calls={self.generator.stats['calls']}",flush=True)
                except Exception as e:
                    errors.append({'question_id':row['question_id'],'error':str(e)})
                    print(f"failed id={row['question_id']}: {e}",flush=True)
                    if isinstance(e,FatalGenerationError):
                        for other in futures: other.cancel()
                        break
        # Collect successful in-flight tasks that finished during fatal shutdown.
        for future,row in futures.items():
            if row['question_id'] not in complete and future.done() and not future.cancelled():
                try: complete[row['question_id']]=future.result()
                except Exception: pass
        dump_json(checkpoint,{str(k):v for k,v in complete.items()})
        dump_json(checkpoint.parent/'run_report.json',{'config':config,'completed':len(complete),'errors':errors,'usage':self.generator.stats,
            'pending_ids':[r['question_id'] for r in rows if r['question_id'] not in complete]})
        if errors: raise GenerationError(f'{len(errors)} unresolved questions; checkpoint retained at {checkpoint}, no submission exported')
        return [complete[r['question_id']] for r in rows]

    def evaluate(self,limit: int | None=None,model: bool=False,workers: int=4,report: Path | None=None) -> dict[str,Any]:
        fit,heldout=split_training(load_json(self.workdir/'QA_train.json'))
        validation=stratified_sample(heldout,limit)
        pipeline=Pipeline(self.workdir,fit)
        groups=defaultdict(Counter)
        baseline=[]
        for row in validation:
            route=pipeline.route(row['question'])
            examples=pipeline._examples(row['question'],route)
            candidate=pipeline._offline_template(row,examples,route)
            baseline.append(candidate)
        def summarize(predictions):
            out=defaultdict(Counter); details=[]
            for row,pred in zip(validation,predictions):
                label=row.get('db','CROSS');key=label+':'+','.join(sorted(referenced_databases(row)))
                actual_dbs=referenced_databases(row);pred_dbs=referenced_databases(pred)
                item={'question_id':row['question_id'],'source_correct':pred.get('db','CROSS')==label,
                      'database_correct':actual_dbs==pred_dbs,'static_valid':validate_record(pred,pipeline.catalog).ok,
                      'normalized_query_match':same_prediction(pred,row),'prediction':pred,'reference':row}
                details.append(item)
                for group in ('overall',key):
                    out[group]['rows']+=1
                    for metric in ('source_correct','database_correct','static_valid','normalized_query_match'): out[group][metric]+=item[metric]
            return {'metrics':{g:{**dict(c),**{k+'_rate':v/c['rows'] for k,v in c.items() if k!='rows'}} for g,c in out.items()},'details':details}
        result={'split_seed':42,'fit_rows':len(fit),'heldout_rows':len(heldout),'evaluated_rows':len(validation),
                'no_question_overlap':not ({question_key(r) for r in fit}&{question_key(r) for r in heldout}),
                'metric_note':'Normalized query match and static validity are proxies, not database execution accuracy.',
                'baseline':summarize(baseline)}
        if model:
            single=pipeline.predict(review=False,workers=workers,rows=validation)
            reviewed=pipeline.predict(review=True,workers=workers,rows=validation)
            result['generated']=summarize(single)
            result['reviewed']=summarize(reviewed)
            result['usage']=pipeline.generator.stats
        if report: dump_json(report,result)
        return result


def validate_submission(workdir: Path, submission: Path) -> dict[str,Any]:
    catalog=load_json(workdir/'catalog.json');records=load_json(submission)
    expected=[r['question_id'] for r in load_json(workdir/'QA_test.json')]
    errors=[]
    if not isinstance(records,list): return {'ok':False,'records':0,'errors':['submission must be a list']}
    ids=[r.get('question_id') if isinstance(r,dict) else None for r in records]
    if ids!=expected: errors.append('question_id list/order differs from QA_test.json')
    for row in records:
        check=validate_record(row,catalog)
        if not check.ok: errors.append(f"id={row.get('question_id') if isinstance(row,dict) else None}: {'; '.join(check.errors)}")
    return {'ok':not errors,'records':len(records),'errors':errors}
