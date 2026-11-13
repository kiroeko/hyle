# 图片与 mmproj

当前完整消费者是 Qwen3.5-2B：Windows／RTX 4070／CUDA，语言权重 BF16、Q8_0、Q4_K_M，普通贪心和随机采样。原始 HF BF16 视觉权重与独立 F16 GGUF mmproj 均已实际运行。CPU、Metal 和其他模型尚未验证完整图片路径；量化 mmproj 只有编码读取机制，未完成整图验证。

## 一次准备，随后复用

先按[引擎指南](inference.md)准备语言模型 `<prepared-dir>`。准备工具只读本地文件，不下载模型。`<model-dir>` 需包含原始 `config.json`、`preprocessor_config.json` 和 tokenizer。

从完整原始 HF checkpoint 借用视觉张量：

```text
python -B scripts/prepare-vision.py --inputs <prepared-dir> --model <model-dir> --output <assets-dir>/vision.tsv
```

已有对应的独立 GGUF mmproj 时：

```text
python -B scripts/prepare-vision.py --inputs <prepared-dir> --model <model-dir> --mmproj <mmproj.gguf> --output <assets-dir>/vision.tsv
```

mmproj 在这里包含视觉编码器与投影头，并非任意线性矩阵。当前 importer 接受 Qwen3-VL merger 布局，核对源模型的层数、维度、patch、merge、位置与归一化配方。当前模型实现不接受 deepstack 条件或视频；换投影数学需要相应模型实现，不能只靠矩阵形状相同加载。

`vision.tsv` 绑定准备后的语言权重来源、视觉角色、物理编码及原文件范围的 SHA256。加载再次核对实际范围；保持原 HF／GGUF 文件可读，不移动或覆盖在途引用。工具保留原编码，不另存整份视觉权重。GGUF 的两个分离时间 patch 核会重排成一个小型 `vision-patch-*.bin`，它也需保留。更换语言模型准备目录或视觉源后重新准备资产，不手改来源身份。

文本前端需要 `transformers`、`tokenizers`、`jinja2`，图片处理额外需要 `Pillow` 和 `numpy`。服务运行不依赖 PyTorch。

## CLI 与服务

```text
python -B -m engine.text.app generate --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --vision <assets-dir>/vision.tsv --image photo.png --text "Describe this picture briefly." --max-output-tokens 128
python -B -m engine.text.app serve --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --backend cuda --vision <assets-dir>/vision.tsv --port 2048 --metrics
```

`--image` 可重复，CLI 将图片按参数顺序放在文字之前。HTTP 可交错文字与图片，三种协议的字段见[接口指南](text-api.md#图片输入)。`--vision` 增加名为 `vision` 的普通执行方案，与原文本方案共享目标权重和端点；图片默认选择它，纯文本仍使用原默认方案。也可在 TOML 的 `[[profiles]]` 中自行声明：

```toml
[[profiles]]
name = "pictures"
draft_count = 0
overlap = false
assets = { "vision.encoder" = "C:/assets/vision.tsv" }
```

多套视觉方案共存时通过 `hyle.execution_profile` 明确选择。当前图片方案不准备 MTP 或设备接力；显式请求这些实现会报错。原文本 MTP 不受此限制。图片请求仍走现有接纳、会话、取消、交付与退休链，不另建执行器。

## 模块边界与速度

| 职责 | 所属模块 |
|---|---|
| 三种 API 图片块、内联数据限额 | `engine/text/media.py`；不解释投影数学 |
| 来源绑定、视觉角色与预处理／patch 布局 | `engine/models/qwen35/vision_preparation.py`、`vision_processing.py`；由静态模型目录装配 |
| 不透明提示附件及精确 token 区间 | 公共请求和 token transport；不要求固定投影或 cache 布局 |
| ViT、merge 投影、语言 embedding 替换及多轴位置 | Qwen 模型实现，使用原 Compute／Graph |
| 权重、队列、原 Invocation 与真实退休 | 原端点、权重流、ProgramIo／ProgramCarrier；不新增资源账簿 |

视觉激活独立使用 F32，随后转换为对应语言模型激活格式。浮点投影按矩阵批量计算，避免逐 patch 向量调用。编码结果通过一次端点内复制形成独立不可变输入；不同图片不会覆盖同一个物理输出。分块 prefill 仅交接本块覆盖的特征行，decode 不再执行视觉编码器。

视觉程序按实际网格和输出类型准备，最多缓存两个形状；首次图片仍包含构图与准备开销。该缓存不按模型名称、语言或 GPU 指纹放行。不同后端通过原 Graph 的算子、编码与资源能力选路；当前 CUDA 的实际通过不能推定 CPU／Metal 全链路通过。

一次性 CLI 仍承担 Python tokenizer 构造、绑定核验、模型与视觉冷准备。常驻服务复用这些资源。缓存后视觉编码的局部计时不代表完整 CLI 耗时，也不能代替加载至退出的对照。

数值对照使用本地 Transformers 5.3.0 的同精度视觉实现；源配方参考 [Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。独立 mmproj 的角色布局参考本地 llama.cpp `23b0202a1` 的 [mtmd](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd) 与转换器。上述参考不统一各模型的投影数学或多轴状态。

无需下载的检查：

```text
python -B -m unittest tests.tooling.vision_pipeline_test tests.tooling.inference_architecture_boundaries_test
python -B -m tests.tooling.vision_local_integration_test --engine <build-dir>/bin/hyle.exe --inputs <prepared-dir> --tokenizer-dir <model-dir> --vision <assets-dir>/vision.tsv --port 2048 --metrics
```

第二条需要已准备的本地 Qwen3.5-2B 与 CUDA。它检查三种 API 及流、多图、双用户、图片续轮与完整重放一致、随机采样，以及非法输入之后继续服务。
