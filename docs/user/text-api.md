# 文本 CLI 与 HTTP 接口

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

三种协议的 Python SDK 均可通过 `extra_body={"hyle": {...}}` 传入。标准 `temperature`／`top_p`，以及 Messages 的 `top_k`，只能匹配已声明的采样程序，不会在请求中编译新程序。草稿上限只能收紧准备深度；上限零不把 MTP 方案变成普通方案。CLI 对应选项为 `--execution-profile NAME --sampling-profile NAME --request-seed N`。

## Chat Completions 与 Messages

同一个 `serve` 命令同时提供下列路由。它们接入相同的文本准备、请求接纳、采样选择、KV 预算和退休机制；协议适配器不创建独立模型进程或调度器。

| 协议 | 路由与文本范围 | 流式结束 |
|---|---|---|
| [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) | `POST /v1/chat/completions`；developer／system／user／assistant 文本消息，字符串或 `type: text` 块；`n=1` | data-only SSE，`chat.completion.chunk`，最后 `data: [DONE]` |
| [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create) | `POST /v1/messages`；user／assistant 文本消息、顶层 `system`，字符串或 `type: text` 块；必填 `max_tokens` | `message_start`、内容块事件、`message_delta`、`message_stop` |

Chat 可用 `max_completion_tokens` 或 `max_tokens` 指定额度，不能同时提供两个非空值；省略时默认 256。
`stream_options.include_usage: true` 在结束前增加一个 `choices: []` 的用量 chunk。
`seed` 接受有符号 64 位整数并按位映射到引擎的 64 位 seed；它不能与 `hyle.seed` 冲突。
Messages 以 assistant 消息结束时使用模板的 `continue_final_message`，续写该前缀；输出仅包含新生成的部分。
模板无法渲染所给角色或前缀时返回 400。

Chat 的 `stop` 支持字符串或最多四个字符串，Messages 使用 `stop_sequences` 数组。
停止串即使跨解码块也不会出现在输出中；匹配后取消剩余工作并确认原生退休，再发送成功终止事件。
自然停止对应 Chat 的 `finish_reason: stop`、Messages 的 `stop_reason: end_turn`；额度截断对应 `length`／`max_tokens`。
Messages 的自定义停止返回 `stop_sequence` 及匹配的字符串。用量按实际消费的 token 计，包含 EOS 或形成停止串的 token，不是对显示文本重新分词。

两种接口均为无状态完整历史请求。调用方每次提交 `messages`；不提供 Chat 存储／查询、跨协议响应 ID 或 Messages 线程接口。
Chat `store: true` 会报错。需要保存及续接时使用 Responses。

```python
from openai import OpenAI
import anthropic

chat = OpenAI(base_url="http://127.0.0.1:2048/v1", api_key="local")
result = chat.chat.completions.create(
    model="hyle", messages=[{"role": "user", "content": "你好"}], max_completion_tokens=64)
print(result.choices[0].message.content)

messages = anthropic.Anthropic(base_url="http://127.0.0.1:2048", api_key="local")
with messages.messages.stream(
    model="hyle", messages=[{"role": "user", "content": "你好"}], max_tokens=64) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

SDK 是可选客户端依赖，服务端不依赖它们。上述 key 只满足客户端构造要求，本地服务没有认证。
格式分别参照 [Chat SSE](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)
和 [Messages SSE](https://platform.claude.com/docs/en/build-with-claude/streaming)。

## Responses 与进程内会话

`POST /v1/responses` 支持文本 `input`、`instructions`、`max_output_tokens`、`stream`、`store`、`previous_response_id` 及上述选择。`model` 必须匹配服务名，默认是 `hyle`。输入支持字符串或 system／user／assistant 文本消息；输出为文本响应或 SSE。其他字段明确拒绝。

服务默认 `store: true`，仅保存在当前进程，受 `sessions.capacity` 和空闲 TTL 限制。`store: false` 不保存新响应。续接时只传新输入和先前响应 ID：

```json
{"model":"hyle","previous_response_id":"resp_...","input":"再给一个例子。","max_output_tokens":64}
```

`instructions` 仅用于当前响应，不自动继承。旧响应可分叉；只有一个分支能借用原生会话，其他分支重建自己的输入。字段语义参照[官方 Responses 合同](https://developers.openai.com/api/reference/resources/responses/methods/create)。

`GET /v1/responses/{id}` 查询保留结果，`DELETE /v1/responses/{id}` 删除该记录并等待它持有的原生会话退休；删除返回[标准删除对象](https://developers.openai.com/api/reference/resources/responses/methods/delete)。删除父记录不删除已有子响应。未知、过期或已淘汰 ID 返回 404。没有跨服务重启的响应／KV 恢复。

响应记录数、原生 KV 驻留上限和批内活动序列是三个独立预算。KV 满时先淘汰空闲原生会话，保留受限历史；全部忙时在有限接纳队列等待。原生终止确认前不复用预算，超时或退休失败会使会话服务失败关闭。TTL 也在没有新请求时回收空闲会话。

满足相同方案、采样配置和实际 token 前缀时，续轮复用原会话的完整状态、pending 和 RNG。Jinja 改写历史前缀、改变配置、KV 淘汰或分叉时重建输入，不伪装缓存命中。保存的原始 token 与模板标记用于可追加模板；显示文本仍独立解码。重建不等于恢复完整 RNG 检查点。

请求初始随机地址由64位 seed、32位逻辑 stream 和32位 draw 分别编码，越界报错；物理槽位、批次位置和请求编号不参与。文本请求的逻辑 stream 为零，未传 seed 时使用配置默认值。相同 seed 在相同数学路径下重放相同随机地址；批形状改变浮点计算时不承诺输出 ID 逐位一致。续轮继续原游标，分叉重放重新开始，不伪装成 RNG 检查点恢复。

响应 `metadata` 报告 `hyle.execution_profile`、`hyle.sampling_profile` 和 `hyle.context`（`new`／`continued`／`replayed`）。`usage.input_tokens_details.cached_tokens` 只计已确认的会话续接 token；原生前缀目录的额外复用不计入该字段。

## 前缀共享与交付

`prefix_cache.entries` 控制模型原状态预算内的完整前缀目录，不另造 KV 银行。前缀包含 KV、卷积、递归等模型完整续算状态，修改继续使用原 COW 与退休机制。默认 `shared: false`，服务为每条会话分配独立共享域；客户端不能提交共享域 ID。`shared: true` 只适合一个可信共享域，不提供租户认证。执行方案和采样数学仍分别限定复用。

Responses SSE 按[官方事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)交付文本和终止事件。达到额度返回 `response.incomplete`；调用方应处理 `response.completed`、`response.incomplete` 和 `error`。旧版 SDK 的 `responses.stream(...).get_final_response()` 可能只接受 completed；可用 `responses.create(..., stream=True)` 处理截断。

无效选择在流开始前返回 400；模型／传输失败返回服务错误，流已开始则交付 `error`。断连或写入失败会取消未完成生成并等待原生终止。TCP 半关闭与优雅完全关闭不能仅靠读 EOF 区分，后者有时需等写入失败。服务关闭先唤醒接纳及连接处理线程，再退休保留会话并关闭模型进程。

本机严格 SDK 检查覆盖三方案并发、可追加模板下的真实 KV 续接、默认模板重建、分叉、查询／删除、非法选择隔离、SSE 和断连后的继续服务。
三种协议支持下列图片输入；工具调用、结构化输出和专门的 thinking 协议字段尚未实现，遇到这些字段明确拒绝。
当前也没有多模型／多量化同时接纳或 KV 磁盘恢复。

## 图片输入

启动时增加 `--vision <vision.tsv>`，或在具名方案的 `assets` 中指定 `"vision.encoder"`。准备与完整命令见[图片与 mmproj](vision.md)。服务复用同一个模型进程；图片请求选择已准备的普通视觉方案，纯文本保留原方案。图片解码还需要 Pillow 与 NumPy。

| 协议 | user 消息中的图片块 |
|---|---|
| Responses | `{"type":"input_image","image_url":"data:image/png;base64,...","detail":"auto"}` |
| Chat Completions | `{"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"auto"}}` |
| Anthropic Messages | `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}` |

文本与图片按块顺序进入模板；支持多图、SSE 和 Responses 历史。只接受内联 base64 的 PNG、JPEG、WebP 与单帧 GIF；远程 URL、file ID、动画和视频会报错。`detail` 为 `auto`／`high`／`low`，实际分辨率同时受源预处理配方和剩余上下文限制。单图编码数据最多 16 MiB，解码最多 32M 像素，HTTP 请求总量最多 32 MiB。没有视觉方案或模板丢失图片占位时返回 400。

图片会话续接同时核对实际 token 前缀及图片载荷身份。默认 Qwen 模板可能改写历史的空 thinking 块或空格，此时完整重放；不会伪报 KV 命中。图片方案关闭仅按 token 匹配的共享前缀目录；完整驻留会话和宿主检查点仍携带 KV、递归状态、pending、RNG 及多轴位置偏移。
