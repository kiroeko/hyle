# CPU 自研精确域：一次有限批量性能裁决

<!-- MODULE REVIEW: CANDIDATE -->

- owner `compute_provider_native`；baseline `db77f16fa`（完整identity由Git导出），Windows canonical。
  SI/MN/TC/TM/AS/CP、ND-R1及RN1/RN4已实际关闭，两个fixture skip也已两平台补跑通过。
- trigger：用户要求每个有成熟替代的CPU自研算子给出实际去留依据，不能永久依靠Native1000后备。
  本卡只接续尚未执行的ND3；不复活旧M5/四路线队列，GPU暂停，SIMD本体和原Torch/ds4/标准借用保留。
- scope：现存generic Native数学/存储族：F32矩阵及物理转置；三floating norm/合法Unary/Binary/
  Reduce/Select；仅保留的F16 SiLU gate；F32/I32 Float32-phase RoPE；Softmax/log/已有mask；
  equal-head attention；Gather/Assign/F32 Add、Concat/Copy/Fill和五floating Cast。
  已退役域不计时、不重写；private/whole-domain/prepared-only约束按现有具名证据单列，不 blanket UNIQUE。
- workload：本地Qwen3.5-2B已核实hidden2048的token/小batch数学片段为primary，rows1与rows32为
  开发/holdout规模；矩阵沿既有1024投影及实际row-major/physical[N,K]转置布局。
  存储/索引/布局为同public task的正交数值管线。只生成一个有界catalog，最多282个同合同case，
  没有consumer需求的域不制造timing。representative不是所有shape/dtype/模型的WIN外推。
- peer：同一已资格化CPU安装SDK的普通默认成熟程序；实际记录provider/implementation/version，
  Native fallback不能冒充peer。Math、storage、alias、checked、资源、线程与完成边界逐侧核对；
  RoPE Float32与Double phase不混比；原库成熟region保留，不能拿拆片leaf比较签完整region收益。
- binding/cost：两侧相同public Context/Queue/Prepared/Invocation/Completion，共享不可变输入与
  同一mutable输出，无alias/并发；冷prepare/分配/first touch不计入steady，另记原cold资源事实。
  steady包含每次输入packing/转换、原private Tensor/库分配、kernel、output materialization、
  检查和真实wait，不遗漏必要成本。没有热选路或用户timing资料。
- 交付/仪器：复用固定产品775b2bed的已验收static SDK，只fresh构建私有C++比较消费者及统一
  controls，绑定SDK/libs/runtime hashes和同源码peer；不再fresh编译全库。W1-WIN/closure-perf-v1、
  单P核/既有priority与电源合同；只读probe，不改用户电源/服务，DLL邻exe/隐藏子进程，j2重型串行。
- budget：接续原ND3未使用采样额度，采样总额2小时，n15起/最多31有效配对，算法优化0。
  一批无计时数值门、同workload A/A、AB/BA与分配次序预检；预检失败先查源码和成本，最多一次
  具名fixture/测量纠正30分钟，不追加姿势或另起一轮采样预算。统一runner/stat核心不改。
- disposition：资格化方向LOSS即退役相应ordinary自研域；无完整优势的PARITY/INCONCLUSIVE选成熟，
  不伪造LOSS/WIN。WIN只固定适用硬件/软件/模型/精度/shape/layout规则；真实unique资源域保留其
  准确边界，不把此理由扩成整个目录。结果先固定有限产品FIX，再进入AUDITED修改和批量验收。
- current：AUDITED仅冻结下列benchmark资格/采样FIX；不授权尚未定界的产品删除、
  API、默认组装器或其它CLOSED owner改动。产品FIX若涉及其它owner，必须据具体结果独立绑定。
- terminal：适用gate未运行不PASS；不可比不签SOTA。预算止后按实际证据裁决，不无限研究。
  产品变更后一次所选factor/实际Direct+Graph与必要两平台三交付、closure/docs/selector/统一LOC；
  无产品变更不重建。完成本轮最终CPU路由/去留后才审核整个goal完成，不能以本计划代签。

## 有限审计与 FIX

| ID | 已固定事实 / 允许动作 | 停止 |
|---|---|---|
| ND3-P | 278 场景普通 mature 数值预检通过；8 个实际同实现对照中206项满足既定范围。冻结这些项的统一 P1 合同，原产品与普通 API 不变 | 同 SDK/源码/loaded code 资格复用；一次n15、最多预定n31，优化0；不得重测72项寻找PASS |
| ND3-U | 72项A/A未资格，整体FAIL原样保留；当前没有正式方向 | 不签LOSS/SOTA；按无完整优势优先成熟的去留政策审计，具名独有能力单列 |
| ND3-D | P1与独有域审计完成后，据实际结果固定有限产品删除/默认规则FIX及必要其它owner | 此行只规定输出，不提前授权未列明的产品变更 |

源级时序：每臂 cold Context/Queue/prepare/invocation，caller resident 输入与共同输出；steady 为
实际 submit 到 Completion::wait，所有 provider 内部 Tensor/packing/转换/输出物化仍在计时内。
Native caller workspace 与成熟原私有状态不同，但普通无限预算串行合同相同；不能由此签 strict
资源或融合 region 的可比性。源码入口是 `native_domain_bench.cc::Session::run/measure`、
Native `build_plan/submit`、ggml prepared graph、oneDNN primitive 与 ATen `torch_cpu_program`。
Native O(elements)/矩阵乘与各原库程序；Native热私有分配0、成熟热原库分配/转换视原程序计费，
公开 checked admission/真实完成两臂均未删减。配置1线程，不把旧scalar ggml当优化peer。

[完整前置证据](../../../../evidence/performance/cpu-native-domain-preflight/README.md)记录数值/对照、
一次具名测量纠正、无效并发尝试及完整72项失格；它不是正式性能WIN/LOSS或新产品完成。
统一合同仅206项，原278 catalog合同另存，C++仍执行全部数值和固定catalog；额外行不参加产品门。
rows1/32的token/small batch与正交存储消费只绑定精确op/kind/dtype/shape/layout/资源/平台版本，
无图融合或整个模型频率外推。SIMD基础与原版 borrowing 不在删除候选内。

## 固定采样候选

- benchmark/合同/资格输入：`481fc3ea2bd2bc9ab50e6b4185c7daa64111ddb3`；产品数学固定 `775b2bedace019de423edc5d8a6477772b8ef3ec` 的已验收SDK。
- 52个SDK安装输入、8个运行DLL和244个同源码peer输入由资格器核对；每项实际mature身份另记。
- 没有产品/API/layout/算法修改；候选后仅状态/证据，不因此重编译产品或重跑旧验收。
- 正式统一采样尚未运行；只能在同loaded code与SDK资格复用成立后开始，适用control未PASS不能签方向。
