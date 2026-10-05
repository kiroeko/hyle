# CPU自研算子：逐精确域与成熟Provider裁决

<!-- MODULE REVIEW: SCOPED -->

- baseline：`59f68e326`，Windows canonical。用户要求：有成熟替代的所有自研算子都要裁决，
  不能因为低优先级后备就永久保留；只有唯一实现的精确域才直接保留。SIMD基础不删除。
- owner：后续涉及实际产品修改的Native/CPU route按当前registry绑定；先完成CPU标准构建卡。
- scope：现存Native全部payload数学域（MatMul、norm、activation/gate、reduce/softmax、
  attention/RoPE、index/transform/cast），相对于oneDNN、ggml、已接Torch成熟域逐项核对。
  TorchAO/ds4原版数学、ATen调用和checked生命周期adapter与自研数学分开。
- 证据：D4的宽泛KEEP_RESIDUAL不是新裁决；当前默认先oneDNN100、ggml200、Native1000，
  优先级本身不证明胜负或保留依据。旧原子/完整片段性能只能在身份与合同匹配时复用。

## 顺序与停止

1. 从实际prepare/propose条件批量生成精确域表：op、dtype/storage/block、shape/layout、
   数值/alias/failure/资源/线程、平台/ISA/软件、模型画像/端口角色；列出具名成熟候选和缺口。
2. 没有成熟同合同候选的域记录UNIQUE_IMPLEMENTATION及源码证据；其余全部进入DECISION_REQUIRED。
   无证据不是WIN，测试通过也不是性能领先，不将宽泛fallback记录当免审理由。
3. 有有效旧同合同证据则直接复用；其余按相同机制批量固定有限代表/holdout消费、原数值门和
   必要转换/完成/资源成本，在不可变候选上一次适用P1；预检失败先查源码，不无限测量或优化。
4. 按严格方向和工程门固定KEEP_WIN/RETIRE_OVERLAP/KEEP_UNIQUE/明确测量阻塞；
   路由规则表达实际dtype/量化、模型/硬件/软件/shape条件，准备时绑定，执行不比赛。
5. 先固定有限删除/路由FIX，再修改产品；一批构建/正确性/必要三交付后终审。SIMD基础保持，
   只有不再有独立用途的算子专用扩张另有具名证据才随域裁决。

本卡目前只是SCOPED审计，尚未测新比较、删算子或签WIN。具体有限样本/成本与FIX须AUDITED
后才开始，禁止生成逐算子无界benchmark目录；在用户确认的既有矩阵上批量处理。
