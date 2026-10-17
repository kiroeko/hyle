# Qwen3.5 2B 命令消费者预检

2026-10-13，HYLE 输入为 `1a15794c0`。用户希望用本地模型建立完整命令消费者，并处理前次
十项机制缺口；[范围与工作量](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/project/qwen35-2b-command-consumer-scope-2026-10.md)
保存具体数学、生命周期、所有 G 项的回映及估计条件。

本目录是**外部参考执行**与模型输入证据，不是 HYLE 整模型 verdict，也不是正式 engine
性能。没有修改 Downloads 模型、HYLE 产品/registry/CMake、第三方锁，没有构建全库。

## 实际输入与执行

两次成功运行都按既有 inventory 完整核验原13文件的size/SHA256；精确 source ID 为
`4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`。
CPU/GPU 实查为 i9-13900K/RTX4070，驱动617.14。只加载本地语言模型320个tensor，tied head
复用 embedding，text weight bytes为3,763,655,360。vision/MTP 不运行。

| 记录 | 实际结果 | 范围 |
|---|---|---|
| [reference-generation.json](reference-generation.json) | 输入23token，生成2token（`5`与结束token） | arithmetic提示；完整24层、初始prefill、1步cached decode |
| [reference-generation-32.json](reference-generation-32.json) | 输入21token，生成32token，说明photosynthesis | 完整24层、初始prefill、31步cached decode；达到长度上限，不能称自然EOS |

每份记录包括完整 prompt/generated IDs、生成文本、320 tensor的shape/dtype/bytes、final cache
shape/dtype/bytes、工具版本、script/模型源码SHA256、步骤诊断时间、GPU峰值allocated/reserved。
两份记录不代表多请求、长上下文、其他平台，时间包含当前参考公式和观测检查，不签token/s。

运行使用已安装Torch2.8.0+cu129、Transformers5.3.0、Safetensors0.8.0、Tokenizers0.22.2，
Torch eager attention、fallback GDN/conv，显式关闭相关reduced-precision reduction与TF32。
执行前强制本地/离线环境，未安装额外依赖或下载模型。

首次诊断脚本在模型加载后因Transformers的chat-template返回BatchEncoding而读取`.shape`
失败，未执行forward；修正为明确`return_dict=True`取`input_ids`后完成上面两份实际记录。
这是预检consumer接线错误，不计HYLE产品失败，也不改原模型/数值合同。

## 重跑参考命令

在HYLE根目录用已有固定Python环境运行，输出路径须不存在：

```powershell
python evidence/correctness/qwen35-cli-preflight/run_reference.py `
  --model "C:\Users\kiro\Downloads\Qwen3.5-2B" `
  --output .scratch/qwen35-cli-preflight/reference-new.json `
  --prompt "Please calculate 2 + 3. Answer briefly." --max-new-tokens 24
```

[run_reference.py](run_reference.py)只为独立reference，不调用HYLE；不得换标签宣称HYLE CLI完成。
重跑时需实际GPU可用，权重加1GiB预检余量不表示任意prefill scratch/并发/上下文容量保证。
HYLE命令的拟议对象序列、验收与当前未完成状态见范围文档。
