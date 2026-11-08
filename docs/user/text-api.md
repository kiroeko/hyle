# 文本 CLI 与 Responses 接口

文本前端使用本地 Hugging Face tokenizer 和模型提供的 Jinja 模板，将 token 请求交给常驻的 Hyle 模型进程。准备目录必须与 tokenizer 属于同一模型；前端检查 token ID 范围，但不能证明两套文件的来源相同。模型声明的结束 ID 与本地 tokenizer／模型配置中的结束 ID 共同作为请求停止条件。前端只读取本地文件，不下载模型。Python 环境需要 `transformers`、`tokenizers` 和 `jinja2`。

例如，在已构建 Hyle 并准备本地 Qwen3.5-2B 资产后：

```text
python -B -m engine.text.app generate --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --text "1 + 1 = ?" --max-output-tokens 16
python -B -m engine.text.app serve --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --port 2048 --metrics
```

Windows 使用 `hyle.exe`；其他平台的文本服务尚未验证。`--template <file>` 可显式选择本地 Jinja 模板。服务默认监听 `127.0.0.1:2048`，指标位于 `/metrics`，健康检查位于 `/health`。其仅为本地服务，不提供访问控制。

`POST /v1/responses` 要求 `model` 与服务加载的模型名一致（服务默认模型名为 `hyle`），支持 `input`、`instructions`、`max_output_tokens`、`stream` 和 `store: false`。输入支持字符串或 system、user、assistant 文本消息；输出支持非流式文本及 SSE 文本事件。不支持的字段会被拒绝。流式事件参照 [OpenAI Responses 事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

本机使用 OpenAI Python SDK 2.29.0 的严格响应校验，已检查 Q4 普通和 BF16 自动 MTP 的同步与原始流式文本。原始流的调用方应分别处理 `response.completed`、`response.incomplete` 和 `error`；达到输出上限时返回 `response.incomplete`。该 SDK 版本的 `responses.stream(...).get_final_response()` 只接受 `response.completed`，遇到不完整响应会抛异常；需要处理截断时可使用 `responses.create(..., stream=True)` 读取终止事件。

无效请求在非流式响应中返回 HTTP 400；如果流式响应已开始，错误通过 `error` 事件交付。模型执行或传输失败按服务错误处理；单个请求被拒绝后，其他请求仍可使用健康的模型进程。

连接重置或输出写入失败时，流式和非流式请求都会请求取消；非流式生成期间定期检查连接。客户端可以在发送请求后关闭写入方向，继续接收响应；TCP 的读取 EOF 无法区分这种半关闭和优雅的完全关闭，后者可能要等后续写入失败才被发现。取消本身不表示设备工作已经退休。有序关闭服务时，监听结束后会断开活动连接，等待连接处理线程退出，再等待模型进程确认关闭；已开始的流可能因此中断。

服务复用一个模型进程，可接收多个请求；请求在运行期间可进入已释放的会话槽位，容量用尽时排队。不提供跨 HTTP 请求持久 KV 会话，每次请求须自行提供对话上下文。当前不支持工具调用、图片、服务端保存响应、逐请求采样配置或其他协议。采样设置在服务启动时指定；显式设置 `--temperature`、`--top-k`、`--top-p` 或 `--seed` 时须同时指定 `--sampling random`。固定种子不保证不同并发调度下的输出逐项相同：采样流由会话槽位区分，实际批次也可能变化。已观察到同一输入在不同的双请求 prefill／调度下随机输出分叉，具体原因尚未归因。

默认配置允许两个活跃序列，自动 MTP 关闭。当 `--max-sessions 1 --active-sequences 1`、CUDA、贪心且制品符合条件时，自动选择 MTP 和制品推荐的草稿深度；其他条件继续使用普通生成。`--draft-count 0` 关闭 MTP，正数显式指定草稿深度，`--draft-vocabulary` 和 `--draft-head` 可指定本地制品；这两个选项始终将参数视为文件路径，包括名为 `auto`、`full` 和 `original` 的文件。自动选择会将推荐深度限制在上下文容量内。

单会话 CUDA 贪心生成及普通随机采样在准备阶段按能力选择设备接力；多会话仍使用原来的默认路径。可用 `--overlap` 或 `--no-overlap` 显式选择；接力需要额外的首次图准备，短请求的完整进程时间可能增加。
