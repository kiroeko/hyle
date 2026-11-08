# 文本 CLI 与 Responses 接口

文本前端使用本地 Hugging Face tokenizer 和模型提供的 Jinja 模板，将 token 请求交给常驻的 Hyle 模型进程。准备目录必须与 tokenizer 属于同一模型；前端检查 token ID 范围，但不能证明两套文件的来源相同。前端只读取本地文件，不下载模型。Python 环境需要 `transformers`、`tokenizers` 和 `jinja2`。

例如，在已构建 Hyle 并准备本地 Qwen3.5-2B 资产后：

```text
python -B -m engine.text.app generate --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --text "1 + 1 = ?" --max-output-tokens 16
python -B -m engine.text.app serve --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --port 2048 --metrics
```

Windows 使用 `hyle.exe`；其他平台的文本服务尚未验证。`--template <file>` 可显式选择本地 Jinja 模板。服务默认监听 `127.0.0.1:2048`，指标位于 `/metrics`，健康检查位于 `/health`。其仅为本地服务，不提供访问控制。

`POST /v1/responses` 支持 `model`（默认为 `hyle`）、`input`、`instructions`、`max_output_tokens`、`stream` 和 `store: false`。输入支持字符串或 system、user、assistant 文本消息；输出支持非流式文本及 SSE 文本事件。不支持的字段会被拒绝。流式事件参照 [OpenAI Responses 事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

服务复用一个模型进程，可接收多个请求，忙时排队并按有限批次运行；不提供跨请求持久 KV 会话或连续合批。每次请求须自行提供对话上下文。当前不支持工具调用、图片、服务端保存响应、逐请求采样配置或其他协议。采样设置在服务启动时指定；新入口默认关闭 MTP，尚未做与旧入口默认策略等价的性能验收。
