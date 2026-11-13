# 文本 CLI 与 Responses 接口

文本前端使用本地 Hugging Face tokenizer 和 Jinja 模板，将 token 请求交给一个常驻模型进程。源模型准备时生成 `tokenizer-binding.tsv`，绑定权重来源、词表容量与原 tokenizer 的实际分词语义。文本入口在启动模型进程前核验，ID 范围相同但映射、分词规则或特殊 token 不同也会拒绝。没有绑定的旧目录须用原模型目录重新准备；加载入口不自动认领当次传入的 tokenizer。模型和 tokenizer 声明的结束 ID 共同作为停止条件。前端不下载模型。Python 环境需要 `transformers`、`tokenizers` 和 `jinja2`；TOML 配置使用 `tomllib`，旧 Python 可安装 `tomli`。

```text
python -B -m engine.text.app generate --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --text "1 + 1 = ?" --max-output-tokens 16
python -B -m engine.text.app serve --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --port 2048 --metrics
```

服务默认监听 `127.0.0.1:2048`，指标位于 `/metrics`，健康检查位于 `/health`。它是本地服务，没有访问控制。本阶段实际验证 Windows／CUDA／本地 Qwen3.5-2B；其他平台尚未实际验证完整服务。`--template FILE` 可选择本地 Jinja 模板。

聊天模板和临时 padding／truncation 不属于 tokenizer 语义身份。模板仍参与草稿选行制品的配方身份；换模板可以生成另一套不可变制品，不必更换模型绑定。

## 冷期配置与请求选择

一个进程加载一个权重来源。多个执行方案共享原端点和不可变目标／MTP 权重，各自持有程序、草稿映射及完整状态。下面的 `service.toml` 声明全头 K1、子集头 K3 和普通生成；随机采样程序独立声明：

```toml
[default_profile]
draft_count = 1

[[profiles]]
name = "subset"
draft_count = 3
generate = { languages = "zh,en" }

[[profiles]]
name = "plain"
draft_count = 0

[[samplers]]
name = "chat"
mode = "random"
temperature = 0.8
top_k = 40
top_p = 0.95

[sessions]
capacity = 128
kv_capacity = 3
ttl_seconds = 900
max_waiting = 64

[prefix_cache]
entries = 3
shared = false
```

```text
python -B -m engine.text.app serve --engine <hyle.exe> --inputs <prepared-dir-with-MTP> --tokenizer-dir <model-dir> --backend cuda --max-sessions 3 --active-sequences 2 --config service.toml --asset-cache <asset-store> --port 2048 --metrics
```

`default_profile` 覆盖默认方案的启动设置，`profiles` 和 `samplers` 增加具名准备。加载后不能替换方案。不同方案分组执行，同方案仍可合批。`generate` 是模型准备器解释的冷期配方；Qwen 支持 `rows` 和逗号分隔的 `languages`，例如 `zh,en,hu`。首次加载生成来源绑定的完整 token 表、草稿 ID 和原编码紧凑头，再原子发布到 `--asset-cache`；重启复用相同配方。省略缓存路径时使用用户缓存目录下的 `hyle/assets`。不会在推理步骤中生成资产。

预先生成的方法见[独立制品工具](inference.md#一次准备并复用)。显式复用时，把方案的 `generate` 换成：

```toml
assets = { "draft.vocabulary" = "C:/assets/mtp-draft-vocabulary.ids", "draft.head" = "C:/assets/mtp-draft-head.tsv" }
```

配置文件中的相对资产路径以配置文件目录为基准。新来源、tokenizer、模板或配方使用新不可变目录，不覆盖在途引用。词表选择是准备策略，不保证任何语言的接受率。

HTTP 用扩展对象 `hyle` 选择已准备程序。该对象只接受 `execution_profile`、`sampling_profile`、`seed` 和 `max_draft_tokens`：

```json
{"model":"hyle","input":"写一段简短问候。","max_output_tokens":64,
 "hyle":{"execution_profile":"plain","sampling_profile":"chat","seed":7}}
```

OpenAI Python SDK 可通过 `extra_body={"hyle": {...}}` 传入。标准 `temperature`／`top_p` 只能匹配已声明的采样程序，不会在请求中编译新程序。草稿上限只能收紧准备深度；上限零不把 MTP 方案变成普通方案。CLI 对应选项为 `--execution-profile NAME --sampling-profile NAME --request-seed N`。

## Responses 与进程内会话

`POST /v1/responses` 支持文本 `input`、`instructions`、`max_output_tokens`、`stream`、`store`、`previous_response_id` 及上述选择。`model` 必须匹配服务名，默认是 `hyle`。输入支持字符串或 system／user／assistant 文本消息；输出为文本响应或 SSE。其他字段明确拒绝。

服务默认 `store: true`，仅保存在当前进程，受 `sessions.capacity` 和空闲 TTL 限制。`store: false` 不保存新响应。续接时只传新输入和先前响应 ID：

```json
{"model":"hyle","previous_response_id":"resp_...","input":"再给一个例子。","max_output_tokens":64}
```

`instructions` 仅用于当前响应，不自动继承。旧响应可分叉；只有一个分支能借用原生会话，其他分支重建自己的输入。字段语义参照[官方 Responses 合同](https://developers.openai.com/api/reference/resources/responses/methods/create)。

`GET /v1/responses/{id}` 查询保留结果，`DELETE /v1/responses/{id}` 删除该记录并等待它持有的原生会话退休；删除返回[标准删除对象](https://developers.openai.com/api/reference/resources/responses/methods/delete)。删除父记录不删除已有子响应。未知、过期或已淘汰 ID 返回 404。没有跨服务重启的响应／KV 恢复。

响应记录数、原生 KV 驻留上限和批内活动序列是三个独立预算。KV 满时先淘汰空闲原生会话，保留受限历史；全部忙时在有限接纳队列等待。原生终止确认前不复用预算，超时或退休失败会使会话服务失败关闭。TTL 也在没有新请求时回收空闲会话。

满足相同方案、采样配置和实际 token 前缀时，续轮复用原会话的完整状态、pending 和 RNG。Jinja 改写历史前缀、改变配置、KV 淘汰或分叉时重建输入，不伪装缓存命中。保存的原始 token 与模板标记用于可追加模板；显示文本仍独立解码。重建不等于恢复完整 RNG 检查点，固定 seed 也不保证跨槽位／批形状的输出逐项一致。

响应 `metadata` 报告 `hyle.execution_profile`、`hyle.sampling_profile` 和 `hyle.context`（`new`／`continued`／`replayed`）。`usage.input_tokens_details.cached_tokens` 只计已确认的会话续接 token；原生前缀目录的额外复用不计入该字段。

## 前缀共享与交付

`prefix_cache.entries` 控制模型原状态预算内的完整前缀目录，不另造 KV 银行。前缀包含 KV、卷积、递归等模型完整续算状态，修改继续使用原 COW 与退休机制。默认 `shared: false`，服务为每条会话分配独立共享域；客户端不能提交共享域 ID。`shared: true` 只适合一个可信共享域，不提供租户认证。执行方案和采样数学仍分别限定复用。

SSE 按[官方事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)交付文本和终止事件。达到额度返回 `response.incomplete`；调用方应处理 `response.completed`、`response.incomplete` 和 `error`。旧版 SDK 的 `responses.stream(...).get_final_response()` 可能只接受 completed；可用 `responses.create(..., stream=True)` 处理截断。

无效选择在流开始前返回 400；模型／传输失败返回服务错误，流已开始则交付 `error`。断连或写入失败会取消未完成生成并等待原生终止。TCP 半关闭与优雅完全关闭不能仅靠读 EOF 区分，后者有时需等写入失败。服务关闭先唤醒接纳及连接处理线程，再退休保留会话并关闭模型进程。

本机严格 SDK 检查覆盖三方案并发、可追加模板下的真实 KV 续接、默认模板重建、分叉、查询／删除、非法选择隔离、SSE 和断连后的继续服务。当前没有工具调用、图片、Chat／Anthropic 路由、多模型／多量化同时接纳或 KV 磁盘恢复。
