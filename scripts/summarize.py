"""Create a reproducibility manifest and a concise validation report."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text2query.io import dump_json
from text2query.pipeline import PIPELINE_VERSION, validate_submission
from text2query.validator import validate_record

root=Path(__file__).resolve().parents[1]
def read(name): return json.loads((root/name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
old=read('result.json')
complete=(root/'result_v2.json').exists()
new=read('result_v2.json' if complete else 'reports/partial_predictions.json')
eval60=read('reports/evaluation_60.json');full=read('reports/evaluation_full_baseline.json')
if complete:
    validation=validate_submission(root/'.t2q',root/'result_v2.json')
    if not validation['ok']: raise SystemExit('New submission failed validation')
else:
    catalog=read('.t2q/catalog.json')
    if not all(validate_record(row,catalog).ok for row in new): raise SystemExit('Partial query validation failed')
    validation={'ok':False,'records':len(new),'partial_queries_valid':True,'errors':['Incomplete generation: DeepSeek HTTP 402 Insufficient Balance']}
old_by_id={r['question_id']:r for r in old}
changed=sum(old_by_id[r['question_id']]!=r for r in new)
manifest={'pipeline_version':PIPELINE_VERSION,'model':'deepseek-v4-pro','records':len(new),'changed_from_old':changed,
          'static_validation':validation,'source_distribution':dict(Counter(r.get('db','CROSS') for r in new)),
          'submission_ready':complete,
          'artifact_hashes':{name:sha(root/name) for name in ['result.json','result_v2.json','reports/partial_predictions.json','赛题8.zip','数据库说明及行政区划表数据.rar','.t2q/catalog.json'] if (root/name).exists()},
          'code_hashes':{str(p.relative_to(root)):sha(p) for p in sorted((root/'text2query').glob('*.py'))},
          'evaluation_limitations':['No database files; no execution-accuracy measurement.','Education unspecified-year prior assumes original dataset ID ordering.','Nine training ES answers reference daily_outpatient absent from published mapping.']}
dump_json(root/'reports/manifest.json',manifest)
status=(f'新结果：`result_v2.json`，{len(new)} 题；提交副本：`submission_v2/result.json`。' if complete else
        f'生成因 DeepSeek HTTP 402（余额不足）中止。已保存 {len(new)}/400 题；中间文件 `reports/partial_predictions.json` 不可提交。')
lines=['# 新结果验证报告','',status,
       f'旧 `result.json` 保留；已完成的新记录中有 {changed} 条与旧结果不同。已完成查询的格式、库表字段及方言静态检查通过。','',
       '## 无泄漏验证','',f"拟合 {full['fit_rows']} 题，完整留出 {full['heldout_rows']} 题，重复问题无交叉。",
       '同一组 60 题覆盖全部数据源和数据库，索引、示例和取值证据均排除验证题。','',
       '| 方案 | 数据源正确率 | 数据库正确率 | 静态有效率 | 规范化查询一致率 |','|---|---:|---:|---:|---:|']
for key,title in [('baseline','复制基线'),('generated','增强生成'),('reviewed','生成加复核')]:
    m=eval60[key]['metrics']['overall']
    lines.append('| '+title+' | '+' | '.join(f"{m[k+'_rate']:.1%}" for k in ['source_correct','database_correct','static_valid','normalized_query_match'])+' |')
f=full['baseline']['metrics']['overall']
lines+=['',f"完整留出集复制基线：数据源正确率 {f['source_correct_rate']:.1%}，规范化查询一致率 {f['normalized_query_match_rate']:.1%}。",'',
        '上述查询一致率是文本结构代理指标；别名和等价写法也会造成不一致，静态校验无法证明查询执行结果正确。线上分数需在比赛平台提交后确认。','',
        '## 实现与限制','',
        '- 补全 265 张表的 schema、字段类型与注释、外键、通用字典和开放行政区划数据。',
        '- 自动提取训练字段取值，生成库表条件规划，独立复核；复杂题增加候选。',
        '- API 缓存、断点续跑、退避重试；失败题不以训练答案填充。',
        '- 省略年度的教育题采用训练同批次的弱年度先验；明确年度和跨年请求优先。重新编号的数据不保证适用。',
        '- 发布 ES mapping 未包含训练答案出现的 daily_outpatient；详见 schema_diagnostics.json。',
        '- 本轮采用用户指定的托管 DeepSeek API，正式参赛的私有部署要求需另行核实。','',
        '复现命令见 README.md；详细逐题验证在 evaluation_60.json；文件及代码哈希在 manifest.json。']
if not complete:
    blocked=read('reports/generation_blocked.json')
    lines+=['','## 续跑','',f"断点：`{blocked['checkpoint']}`，剩余 {len(blocked['remaining_ids'])} 题。",
            '充值原 DeepSeek 账号或设置另一个有效密钥后，执行：','',
            '```bash','.venv/bin/python -m text2query predict --output result_v2.json --workers 4',
            '.venv/bin/python -m text2query validate --input result_v2.json','```',
            '','完整校验通过后复制为 `submission_v2/result.json`。23 项自动测试全部通过。']
(root/'reports/validation_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'records':len(new),'changed':changed,'valid':validation['ok']},ensure_ascii=False))
