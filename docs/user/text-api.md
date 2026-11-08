# 文本 CLI 与 Responses 接口

文本前端使用本地 Hugging Face tokenizer 和模型提供的 Jinja 模板，将 token 请求交给常驻的 Hyle 模型进程。准备目录必须与 tokenizer 属于同一模型；前端检查 token ID 范围，但不能证明两套文件的来源相同。前端只读取本地文件，不下载模型。Python 环境需要 `transformers`、`tokenizers` 和 `jinja2`。

例如，在已构建 Hyle 并准备本地 Qwen3.5-2B 资产后：

```text
python -B -m engine.text.app generate --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --text "1 + 1 = ?" --max-output-tokens 16
python -B -m engine.text.app serve --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --port 2048 --metrics
```

Windows 使用 `hyle.exe`；其他平台的文本服务尚未验证。`--template <file>` 可显式选择本地 Jinja 模板。服务默认监听 `127.0.0.1:2048`，指标位于 `/metrics`，健康检查位于 `/health`。其仅为本地服务，不提供访问控制。

`POST /v1/responses` 要求 `model` 与服务加载的模型名一致（服务默认模型名为 `hyle`），支持 `input`、`instructions`、`max_output_tokens`、`stream` 和 `store: false`。输入支持字符串或 system、user、assistant 文本消息；输出支持非流式文本及 SSE 文本事件。不支持的字段会被拒绝。流式事件参照 [OpenAI Responses 事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

服务复用一个模型进程，可接收多个请求；请求在运行期间可进入已释放的会话槽位，容量用尽时排队。不提供跨 HTTP 请求持久 KV 会话，每次请求须自行提供对话上下文。当前不支持工具调用、图片、服务端保存响应、逐请求采样配置或其他协议。采样设置在服务启动时指定。

默认配置允许两个活跃序列，自动 MTP 关闭。当 `--max-sessions 1 --active-sequences 1`、CUDA、贪心且制品符合条件时，自动选择 MTP 和制品推荐的草稿深度；其他条件继续使用普通生成。`--draft-count 0` 关闭 MTP，正数显式指定草稿深度，`--draft-vocabulary` 和 `--draft-head` 可指定本地制品；这两个选项始终将参数视为文件路径，包括名为 `auto`、`full` 和 `original` 的文件。自动选择会将推荐深度限制在上下文容量内。

单会话 CUDA 贪心生成在准备阶段按能力选择设备接力；多会话和随机采样仍使用原来的默认路径。可用 `--overlap` 或 `--no-overlap` 显式选择；接力需要额外的首次图准备，短请求的完整进程时间可能增加。
