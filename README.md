# Challenge Text2Query

可复现的多数据源查询生成流程：库表规划 → schema/知识/字典/训练取值检索 → 模型生成 → 独立复核 → 静态校验 → 提交文件。可选择 DeepSeek API 或本地 Ollama。普通题默认三次模型调用，复杂题增加独立候选。不会在模型调用失败时自动复制训练答案。

项目架构、模块职责、完整数据流和运行步骤见 [项目架构与流程](项目架构与流程.md)。

基于线上 869/1087 分反馈的失分分析、代码问题和改进优先级见 [项目改进分析](项目改进分析.md)。

## 安装与运行

Python 3.10+；Ubuntu 未提供 venv 时需安装 `python3-venv`。RAR 解压需 `unrar`、`bsdtar` 或 `unrar-free`，也可以传入已经解压的 SQL 文件。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m text2query prepare --archive 赛题8.zip \
  --administrative 数据库说明及行政区划表数据.rar --workdir .t2q

export T2Q_PROVIDER=deepseek
export T2Q_API_BASE=https://api.deepseek.com
export T2Q_MODEL=deepseek-v4-pro
read -rsp 'DeepSeek API key: ' T2Q_API_KEY
export T2Q_API_KEY

.venv/bin/python -m text2query evaluate --model --limit 60 \
  --report reports/evaluation_60.json --workers 4
.venv/bin/python -m text2query predict --output result_v2.json --workers 4
.venv/bin/python -m text2query validate --input result_v2.json
.venv/bin/python -m unittest discover -s tests -v
unset T2Q_API_KEY
```

使用本地 Ollama 时，先启动 Ollama 服务并下载模型，然后配置本地兼容接口。已执行过上面的 DeepSeek 命令时，需覆盖其地址和模型名：

```bash
ollama pull qwen3:8b
export T2Q_PROVIDER=ollama
export T2Q_API_BASE=http://127.0.0.1:11434/v1
export T2Q_MODEL=qwen3:8b
.venv/bin/python -m text2query evaluate --model --limit 1 \
  --report reports/evaluation_ollama_smoke.json --workers 1
.venv/bin/python -m text2query predict --output result_ollama.json --workers 1
```

`T2Q_PROVIDER=ollama` 不需要 API key；地址和模型名分别默认 `http://127.0.0.1:11434/v1`、`qwen3:8b`。也可用 `T2Q_MODEL` 指定已经下载的其他模型。按本机推理能力调整 `--workers`、`T2Q_TIMEOUT` 和 `T2Q_MAX_TOKENS`。接口用法见 [Ollama 官方兼容接口文档](https://docs.ollama.com/api/openai-compatibility)。

当前工作区 RAR 工具安装在 `.t2q/tools`，需要时可使用：

```bash
export PATH="$PWD/.t2q/tools/usr/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.t2q/tools/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

## 评估及复现

固定种子 42，按数据源和数据库分层留出约 10%；重复问题归入同一侧。索引、训练示例、取值证据均仅由拟合部分构建。`evaluate` 无 `--model` 时只运行完整留出集复制基线；`--model --limit 60` 比较复制基线、增强生成、生成加复核，覆盖各数据源和数据库。删除 `--limit` 可对完整留出集进行模型评估，但会增加调用数量。

报告包含数据源正确率、数据库正确率、静态有效率、规范化查询文本一致率和逐题差异。没有真实数据库，因此这些指标不能证明执行准确率；等价 SQL 也可能因别名、写法差异而不一致。线上成绩需要提交至比赛平台后确认。

`.t2q/cache` 保存无凭据的响应、实际模型标识和 token 用量；`.t2q/runs` 保存条件规划、候选、复核及逐题断点。保持代码、数据和配置可复用缓存；题目集合、训练集合、模型或 schema 变化会生成独立运行目录。`--no-review` 用于消融，`--no-resume` 忽略最终断点但仍可使用模型响应缓存。

环境变量：`T2Q_TIMEOUT` 默认 300 秒，`T2Q_MAX_TOKENS` 默认 16384；截断时最多扩展至 65536。限流/服务暂时失败最多尝试四次。正式预测有未解决题目时不会导出提交文件。`--offline` 仅保留旧复制基线用于比较。

## 产物

预测示例输出为 `result_v2.json`，完整校验通过后另存 `submission_v2/result.json`；标准生成和打包命令不会覆盖根目录的 `result.json`。当前工作区的根目录 `result.json` 与提交副本内容一致，`result_v2.json` 当前不存在。验证报告位于 `reports/`。密钥只从环境变量或关闭回显的标准输入读取，不保存到代码、日志或结果。

发布 schema 与训练答案存在 `daily_outpatient` 字段冲突，记录在诊断报告中；校验仍以发布的 ES mapping 为依据。历史结果由 DeepSeek API 生成；新预测可选择本地 Ollama。赛题的私有部署要求及正式参赛资格需另行核实。

部分教育题完全省略年度。仅在文本已定位教育库、问题未明确年份且没有跨年要求时，使用拟合集中邻近题号的同批次年度作为弱提示（至少五条、占比至少 85%）。它不决定数据源，也不覆盖明确年度；这项数据分布假设不能保证适用于重新编号的数据。

模型响应使用流式传输，`T2Q_TIMEOUT` 同时限制单次生成的总时长与连接闲置时间，避免非流式保活空白造成无限等待。复杂候选的 `max` 推理超时后降到 `high` 重试，实际推理级别记录在缓存中。请求开始、完成与重试事件记录于 `.t2q/cache/events.jsonl`，不含密钥或请求正文。

跨源上下文明确提供 `transportation.vessel.mmsi` 与 AIS `mmsi` 的关联，即使初始检索只命中 PostgreSQL，也可重新路由到船舶登记与公司表。输出公司信息时，可先从 AIS 筛选 mmsi，再在 MySQL 查询公司信息。模型返回 `prediction:null` 或缺少提交字段时，会带着结构错误再次请求完整预测。超时或断流会依次降低推理级别以恢复生成；截断会扩大输出预算，`max` 截断时改用 `high`。

限流与暂时服务错误使用带随机偏移的指数退避，遵守 `Retry-After`（最多等待 120 秒）；`T2Q_RETRY_BASE` 默认 5 秒。早期生成遇到限流后采用 4 个并发；充值续跑确认接口稳定后，从断点恢复为 8 个并发。并发数不影响断点或缓存复用。

规划响应缺少有效数据源或只包含局部 SQL/DSL 时，会附带原响应重新请求完整规划，最多三次。若规划阶段意外返回完整查询，则仅提取路由信息，仍执行后续生成与复核。

HTTP 401/402/403 等不可恢复错误会触发共享熔断，取消尚未开始的题目并收集已经完成的请求。余额不足后充值原账号或设置新的有效密钥，再运行原 `predict` 命令即可续跑；更换同模型密钥不影响已有断点。此时 `reports/partial_predictions.json` 是中间结果，不可作为 400 题提交文件。

生成完整文件后可自动检查并打包：

```bash
.venv/bin/python scripts/finalize.py
```

此脚本验证完整测试 ID 和查询格式，再创建 `submission_v2/result.json`、提交校验报告和更新后的复现 manifest。原始 `result.json` 不覆盖。
