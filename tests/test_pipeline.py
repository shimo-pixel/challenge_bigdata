import json
import io
import urllib.error
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from text2query.generator import FatalGenerationError, GenerationError, OpenAICompatibleGenerator, extract_object
from text2query.pipeline import Pipeline, question_key, split_training, stratified_sample
from text2query.schema import parse_sql, dictionary_statements
from text2query.validator import validate_query, validate_record

CATALOG={'tables':[
 {'source':'MYSQL','database':'a','name':'parent','columns':['id','name']},
 {'source':'MYSQL','database':'a','name':'child','columns':['id','parent_id','result']},
 {'source':'SQLite','database':'disney','name':'voice-actors','columns':['voice-actor','movie']},
 {'source':'POSTGRESQL','database':'','name':'vessel_ais_data','columns':['mmsi','geom','time']},
 {'source':'Elasticsearch','database':'','name':'medical_institutions','columns':['name','location','bed_count','has_icu']}
]}

class SchemaTests(unittest.TestCase):
    def test_quoted_identifier_and_inline_columns(self):
        tables=parse_sql('数据库: disney\nCREATE TABLE "voice-actors" ("voice-actor" TEXT, movie TEXT, FOREIGN KEY(movie) REFERENCES characters(movie_title));','SQLite')
        self.assertEqual(tables[0].name,'voice-actors')
        self.assertEqual(tables[0].columns,['voice-actor','movie'])

    def test_comment_parentheses_and_literal_comma(self):
        tables=parse_sql("CREATE DATABASE a; CREATE TABLE t (id INT, x VARCHAR(20) COMMENT 'units (a,b)', v DECIMAL(8,2), PRIMARY KEY (id)) ENGINE=InnoDB COMMENT='table';",'MYSQL')
        self.assertEqual(tables[0].columns,['id','x','v'])
        self.assertIn("COMMENT='table'",tables[0].ddl)
        ds=dictionary_statements("INSERT INTO `common`.`x` (id,name) VALUES ('1','a,b;test'); INSERT INTO x(id,name) VALUES('2','z');")
        self.assertEqual(len(ds),2)
        self.assertEqual(ds[0]['table'],'x')

class QueryTests(unittest.TestCase):
    def test_nested_column_is_not_checked_against_outer_table(self):
        self.assertTrue(validate_query('MYSQL','SELECT id FROM a.child WHERE parent_id IN (SELECT id FROM a.parent WHERE name=\'x\')',CATALOG).ok)

    def test_unknown_fields_database_and_multiple_statements(self):
        for sql in ['SELECT missing FROM a.child','SELECT id FROM wrong.child','SELECT id FROM child','SELECT id FROM a.child; DELETE FROM a.child']:
            self.assertFalse(validate_query('MYSQL',sql,CATALOG).ok,sql)

    def test_cte_quoted_names_and_correlated_subquery(self):
        self.assertTrue(validate_query('SQLite','SELECT "voice-actor" FROM disney."voice-actors"',CATALOG).ok)
        self.assertTrue(validate_query('MYSQL','WITH x AS (SELECT id FROM a.parent) SELECT id FROM x',CATALOG).ok)
        self.assertTrue(validate_query('MYSQL','SELECT p.name FROM a.parent p WHERE EXISTS (SELECT 1 FROM a.child c WHERE c.parent_id=p.id)',CATALOG).ok)

    def test_real_result_column_and_placeholder(self):
        self.assertTrue(validate_query('MYSQL','SELECT result FROM a.child',CATALOG).ok)
        self.assertTrue(validate_query('POSTGRESQL','SELECT mmsi FROM vessel_ais_data WHERE mmsi IN (result)',CATALOG,True).ok)
        self.assertFalse(validate_query('POSTGRESQL',"SELECT mmsi FROM vessel_ais_data WHERE time='result'",CATALOG,True).ok)

    def test_es_fields_types_and_structure(self):
        self.assertTrue(validate_query('Elasticsearch',json.dumps({'query':{'term':{'has_icu':True}},'_source':['name']}),CATALOG).ok)
        for value in [[],{}, {'query':{'range':{'bed_count':{'gt':'20'}}}}, {'query':{'term':{'has_icu':'yes'}}}, {'_source':['nonexistent']}]:
            self.assertFalse(validate_query('Elasticsearch',json.dumps(value),CATALOG).ok)

    def test_strict_record_keys(self):
        self.assertFalse(validate_record({'db':'MYSQL','query':'SELECT id FROM a.parent','question_id':1,'question':'x'},CATALOG).ok)

class LeakageTests(unittest.TestCase):
    def test_year_prior_only_applies_to_unspecified_year(self):
        from text2query.retrieval import CharTfidf
        p=Pipeline.__new__(Pipeline)
        p.train=[{'question_id':i,'question':'查询教师姓名','db':'MYSQL','query':'SELECT teacher_name FROM education_2025.teacher'} for i in range(80,120)]
        p.row_databases=[{'education_2025'} for r in p.train]
        p.retriever=CharTfidf(r['question'] for r in p.train)
        self.assertEqual(p._hints('查询教师姓名',100)['default_database'],'education_2025')
        self.assertNotIn('default_database',p._hints('查询2024年教师姓名',100))
        self.assertNotIn('default_database',p._hints('查询跨年教师姓名',100))

    def test_duplicate_questions_stay_on_same_side(self):
        rows=[{'question_id':i,'question':f'question {i//2}','db':'MYSQL','query':'SELECT id FROM a.parent'} for i in range(40)]
        fit,heldout=split_training(rows)
        self.assertFalse({question_key(x) for x in fit}&{question_key(x) for x in heldout})
        self.assertEqual(split_training(rows),(fit,heldout))
        self.assertEqual(len(stratified_sample(heldout,2)),2)

    def test_examples_exclude_before_ranking(self):
        p=Pipeline.__new__(Pipeline)
        from text2query.retrieval import CharTfidf
        from collections import defaultdict
        p.train=[{'question_id':i,'question':'相同的示例问题','db':'MYSQL','query':f'SELECT {i}'} for i in range(8)]
        p.retriever=CharTfidf(x['question'] for x in p.train);p.by_db={'MYSQL':set(range(8))};p.by_database=defaultdict(set)
        self.assertEqual(len(p._examples('相同的示例问题','MYSQL',exclude_id=0)),1)
        self.assertFalse(any('question_id' in x for x in p._examples('相同的示例问题','MYSQL',exclude_id=0)))
        self.assertNotEqual(p._examples('相同的示例问题','MYSQL',exclude_id=0)[0]['query'],'SELECT 0')

class CheckpointTests(unittest.TestCase):
    def test_failure_is_checkpointed_and_never_exported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Pipeline.__new__(Pipeline)
            p.workdir=Path(directory);p.fingerprint='test';p.catalog=CATALOG
            p.generator=OpenAICompatibleGenerator()
            p.test=[{'question_id':1,'question':'success'},{'question_id':2,'question':'failure'}]
            def generate(row,*args):
                if row['question_id']==2: raise GenerationError('test transient failure')
                return {'question_id':1,'db':'MYSQL','query':'SELECT id FROM a.parent'}
            p.predict_one=generate
            with self.assertRaises(GenerationError): p.predict(workers=1)
            report=json.loads(next(Path(directory).rglob('run_report.json')).read_text())
            self.assertEqual(report['completed'],1)
            self.assertEqual(report['errors'][0]['question_id'],2)
            p.predict_one=lambda row,*args:{'question_id':row['question_id'],'db':'MYSQL','query':'SELECT id FROM a.parent'}
            output=p.predict(workers=1)
            self.assertEqual([x['question_id'] for x in output],[1,2])

class GeneratorTests(unittest.TestCase):
    def test_local_ollama_uses_keyless_compatible_endpoint(self):
        chunk={'model':'qwen3:8b','choices':[{'delta':{'content':'{"prediction":{"db":"MYSQL","query":"SELECT 1"}}'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':3,'completion_tokens':4}}
        response=io.BytesIO(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n').encode())
        with patch.dict('os.environ',{'T2Q_PROVIDER':'ollama','T2Q_API_BASE':'http://127.0.0.1:11434/v1',
                                   'T2Q_MODEL':'qwen3:8b','T2Q_API_KEY':'','DEEPSEEK_API_KEY':''}), \
             patch('urllib.request.urlopen',return_value=response) as call:
            generator=OpenAICompatibleGenerator()
            self.assertEqual(generator.generate('return json'),{'prediction':{'db':'MYSQL','query':'SELECT 1'}})
            request=call.call_args.args[0]
            self.assertEqual(request.full_url,'http://127.0.0.1:11434/v1/chat/completions')
            self.assertIsNone(request.get_header('Authorization'))
            body=json.loads(request.data)
            self.assertEqual(body['model'],'qwen3:8b')
            self.assertNotIn('thinking',body)
            self.assertEqual(body['response_format'],{'type':'json_object'})
            self.assertTrue(body['stream'])

    def test_partial_dsl_plan_is_retried(self):
        p=Pipeline.__new__(Pipeline);p.generator=Mock()
        answer={'sources':['Elasticsearch'],'databases':[],'tables':['medical_institutions']}
        p.generator.generate.side_effect=[{'type':'geo_bounding_box','field':'location'},answer]
        self.assertEqual(p._generate_plan('return route plan',1),answer)
        self.assertIn('Do not return a partial SQL or DSL clause',p.generator.generate.call_args[0][0])

    def test_planner_query_can_supply_explicit_routing(self):
        p=Pipeline.__new__(Pipeline);p.generator=Mock()
        p.generator.generate.return_value={'prediction':{'db':'MYSQL','query':'SELECT id FROM a.parent'}}
        plan=p._generate_plan('return route plan',1)
        self.assertEqual(plan['sources'],['MYSQL'])
        self.assertEqual(plan['databases'],['a'])

    def test_balance_failure_opens_circuit_and_redacts_credentials(self):
        error=urllib.error.HTTPError('https://api.deepseek.com',402,'balance',{},io.BytesIO(b'{"error":{"message":"Insufficient Balance test-key"}}'))
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'T2Q_API_KEY':'test-key'}),patch('urllib.request.urlopen',side_effect=error) as call:
            g=OpenAICompatibleGenerator(Path(directory))
            with self.assertRaises(FatalGenerationError): g.generate('json one')
            with self.assertRaises(FatalGenerationError): g.generate('json two')
            self.assertEqual(call.call_count,1)
            event=(Path(directory)/'events.jsonl').read_text()
            self.assertNotIn('test-key',event)
            self.assertIn('[REDACTED]',event)

    def test_stream_deadline(self):
        g=OpenAICompatibleGenerator()
        with patch('time.monotonic',return_value=301):
            with self.assertRaises(TimeoutError): g._stream_response(iter([b'data: [DONE]\n']),0)

    def test_null_prediction_is_repaired_by_model(self):
        p=Pipeline.__new__(Pipeline)
        p.generator=Mock()
        answer={'prediction':{'db':'MYSQL','query':'SELECT id FROM a.parent'}}
        p.generator.generate.side_effect=[{'prediction':None},answer]
        self.assertEqual(p._generate_prediction('return json',1),answer)
        self.assertIn('Do not return prediction:null',p.generator.generate.call_args[0][0])

    def test_disabled_generator_fails_without_fallback(self):
        with patch.dict('os.environ',{'T2Q_API_KEY':'','DEEPSEEK_API_KEY':''}):
            with self.assertRaises(GenerationError): OpenAICompatibleGenerator().generate('json')

    def test_json_with_braces_in_sql_literal(self):
        obj={'db':'MYSQL','query':"SELECT '{test}'"}
        self.assertEqual(extract_object('```json\n'+json.dumps(obj)+'\n```'),obj)

    def test_truncation_retry_cache_and_usage(self):
        class Response:
            def __init__(self,data): self.data=data
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                choice=self.data['choices'][0]
                chunk={'choices':[{'delta':{'content':choice['message']['content']},'finish_reason':choice['finish_reason']}],'usage':self.data['usage']}
                return iter([('data: '+json.dumps(chunk)+'\n').encode(),b'data: [DONE]\n'])
        truncated={'choices':[{'finish_reason':'length','message':{'content':'{}'}}],'usage':{'prompt_tokens':10,'completion_tokens':20}}
        good={'choices':[{'finish_reason':'stop','message':{'content':'{"prediction":{"db":"MYSQL","query":"SELECT 1"}}'}}],'usage':{'prompt_tokens':10,'completion_tokens':5}}
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'T2Q_API_KEY':'test-key'}),patch('urllib.request.urlopen',side_effect=[Response(truncated),Response(good)]) as call,patch('time.sleep'):
            g=OpenAICompatibleGenerator(Path(d));obj=g.generate('return json')
            self.assertEqual(g.generate('return json'),obj)
            self.assertEqual(call.call_count,2)
            self.assertEqual(g.stats['cache_hits'],1)
            self.assertEqual(g.stats['completion_tokens'],25)
            for p in Path(d).glob('*.json'): self.assertNotIn('test-key',p.read_text())

if __name__=='__main__': unittest.main()
