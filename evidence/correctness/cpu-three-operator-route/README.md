# CPU 三项最终接缝与清退资格

- 最终产品候选 `119137681ed03f58e3e193a97c969be4e45ccb34`；之后只有文档、状态、测试和证据。
- [施工卡](../../../docs/implementation/tasks/compute/cpu-three-operator-route-closure.md)、
  [实际门及归档身份](receipt.json)、[完整日志/源码consumer/XML/同步身份/selector/LOC](qualification.zip)。
- [F32 fused四格唯一P1](../../../evidence/performance/cpu-fused-sdpa-current/README.md)资格通过，
  Native全LOSS，已经源级撤回，不优化或重测。

## 通过的实际旅程

| 环境与交付 | 普通Direct/Graph精确格 | 默认Native优势格 | strided拒绝/成熟物化 | dynamic/view物化 |
|---|---:|---:|---|---|
| Windows/MSVC source链接public facade | 24 | 17 | PASS | PASS |
| Windows installed static | 24 | 17 | PASS | PASS |
| Windows installed shared | 24 | 17 | PASS | PASS |
| Linux/GCC14 source链接public facade | 24 | 0 | PASS | PASS |
| Linux/GCC14 installed static | 24 | 0 | PASS | PASS |
| Linux/GCC14 installed shared | 24 | 0 | PASS | PASS |

每格两次变输入replay，全部成熟region仍存在；默认精确绑定、9种错配、独立并发槽和close拒绝实际通过。
Linux不继承Windows/MSVC performance资格。source父工程/安装胶水的公开合同不变，复用先前嵌入与package资格；
本轮实际source-linked public facade与installed independent consumer重编运行，不能冒称重跑所有共享工具检查。

两OS owning Native/ggml suites通过，包含F32实际64/256宽度、两layout、B1/B2、GQA、causal/noncausal、
长KV、独立double oracle、NaN/Inf/overflow/zero-scale及fresh replay。Windows58项受影响Graph最终全过
（原55 PASS+三个旧身份断言修正重验），七项Direct模型身份修正与executor通过。
旧断言只放行Native RMS + 正确namespace/version2有效policy；没有放开任意Native后备，没有放宽数值门。
删除后两factor与24格/dynamic再次通过；CPU/Graph其它大矩阵、packing/head原数学未变，复用匹配旧资格。

## 保留的失败与限制

1. 第一轮namespace fixture编译漏写Context namespace已修正，原日志保留。
2. 宽泛CPU sweep在旧身份断言出现失败；已核对进程后停止不受改动影响的巨型完整head循环，记INCOMPLETE。
   已完成项原样保留，失败七项重验；不把原aggregate改写PASS，不重复巨大未变模型目录。
3. Linux第一轮`git -C source`读取到了父目录`/`的Git身份，实际archive没有更新；该轮无效。
   最终按Windows不可变输入1654个文件逐SHA核对、只更新变化文件并删除attention.cc。原FAIL和同步证据保留，
   真正同步后owning/24格/dynamic以及两SDK均通过；不从Linux反向覆盖产品。
4. Linux最初consumer选择GCC13；额外static→custom DSO示例遇到非PIC静态对象链接失败。
   普通static C++ executable及同进程dynamic旅程以GCC14独立重编通过。非PIC静态SDK链接进DSO不签资格，
   需用户显式PIC构建或shared SDK；不是本轮动态Graph能力缺失。产品构建系统没有改。
5. Native SDPA的Native-only、caller-workspace/零额外private、非dense/mask执行档明确撤回。
   普通SDPA schema及成熟ggml/ATen GQA/mask/布局/空keys能力继续；没有模型级或GPU资格扩张。
6. 17格规则只限指定硬件/软件/精度/形状/attrs及串行条件；不输入timing profile，不热搜索，不签未知硬件SOTA。
   Copy/RMS原数学不改，SourceMeta减小按P0验证；没有声称刚重跑17格P1或全Native算子目录。

## 最终裁决

G1/N1/N2/N3通过：原F32 fused接对，17格规则绑定，普通exact strided成熟接替；有真实public consumer的
declared-domain view/broadcast物化保留，未以抽象资源差异无限留下普通重叠后备。自研SDPA整项删除，SIMD/Mem保留。
scope/architecture/correctness/delivery/usability_docs/red_team/engineering_performance(P0+有界P1去留)通过。
终审独立复核源码/同合同、负空间、版本条件、whole-domain消费、三交付差异和无GPU改动。

统一Compute+Graph实现 **61,073行/249文件**，不含测试/vendor/benchmark；self_without_tests **156,300行**。
当前计数下CPU自研运算种类从36减为35（15个family）；Copy包含Mem接缝和声明域物化，不是另造dense复制核。
临时产物清理见[Windows记录](cleanup-windows.json)与[Linux记录](cleanup-linux.json)，合计2,155,055,578字节（约2.01 GiB）；
缓存/旧SDK与不相关用户修改保留。

本轮17格仅与已测默认成熟peer比较，不是oneDNN/ggml/ATen所有重叠域全组合最优结论。
其它CPU域仍有固定成熟优先顺序或原版本有限证据，CPU全重叠域best-of尚未完成。
