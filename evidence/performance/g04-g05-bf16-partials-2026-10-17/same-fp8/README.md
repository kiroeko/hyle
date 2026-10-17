# 同 FP8 驻留输入的真实上游参考

本次替代[异格式 warm 对照的错误实现排名](../comparison-boundary-correction/README.md)。
双方从同一份 device-resident E4M3 K/V 页、BF16 Q 和 metadata 开始，没有完整 BF16 影子
缓存或计时外的 KV 表示转换。原 .5/2 缩放与输入字节不改，输出均为 BF16。

原样参考首先直接实例化 FlashInfer `69ff11fc4954396d98326656dc85debd2223f637` 的 BF16 Q / FP8 KV /
BF16 output 模板，使用原 loader、planner、合法 dispatcher 与 merge。原始归档 SHA256 为
`4fc85abaa6210d6a1773e6011ff572aa0c04b7030644b74f75a2db2dbda472a9`，2576文件编前后逐字不变。
仅用 namespace token 隔离 host/device 符号，避免与产品补丁模板相互覆盖。
实际进入计时的是下面单独标识的同步修正参考，不能把它称为 pristine 计时。

原始参考桥首编通过，17.519秒，仅编译1个小 CUDA TU。archive SHA256 为
`020aef8f5454f5601ffe533f7a1c093a987c87068ea5e047f662831231356670`。
产品直接复用实际 C6 archive `bbf3ec34a9aa0d7916ffc3ebb4a340583946ec3ce2bc33a6c0dc2711256ecec2`，
没有重编基线或修改产品。

## 数值与调用边界

Kscale .5 同样折到 softmax scale；参考的派生 Params 仅提供原上游已识别的 `v_scale=2`。
原 OutputTransform 在 partial BF16 窄化前应用该尺度，产品在 merge 后应用；这表示同一
数学任务，但不声称任意数据/不同 split 的舍入逐位相等。每边都必须对同一个原 FP8 字节
推导的 double oracle 使用同一误差约束、有限值和非均匀输出检查。

primary 为 Q64/128、KV1024/768、D256、GQA8/2、page16、causal 无界；产品和参考都允许
BF16 partial。window 为 Q1/17、同 KV/head/page、D256、causal window16。
window/非等深非causal case 必须分别记录实际 partial 类型，不能
将产品在该域保留的 F32 与参考的 BF16 隐藏成同一内部精度。

普通调用的统一边界为 update→run→checked wait→owning output release。cold 的上传、
分配和 plan/close 单列；不把产品 replay 和参考 normal 比较。每个实现保留自己的合法
KV tile、split、workspace，不强塞产品 KMMA1 到上游，也不以人为过小的 cap 拒绝参考。
记录实际需求与实际预留，不能以 cap 相同推断内存占用相同。

参考必须通过实际数值、readonly/寿命及 memcheck/synccheck/racecheck 后，才能解释该
case 的普通计时。辅助 case 的失败单列，不能用不正确输出作速度排名。原样上游若有
并发缺陷则保留失败，任何必要同步修复另列身份，不能暗带到 pristine 结果。

## 原样参考的实际验证

第一次 harness 在 dispatch 前被公共资源身份检查拒绝；未弱化该检查，改为单 Context、
单 MemoryResource 注册两个 provider，并只在冷端选择 provider instance。热端均经相同
公共 BatchInvocation，原输入、metadata admission、结果租约和 drain 保留。

prefill 加 tail、双 slot 的实际数值验证通过，最大绝对误差0.001090813；本有限 fixture
两边输出逐 word 相等。memcheck、synccheck均为零错误。
但原样上游 FP8 的 racecheck 实际报告64条 warnings、进程 exit87，耗时49.406秒。
尚未对这条有警告的参考进行普通计时，不能拿数值通过或编译通过冒充并发安全。
具体竞争归属、必要同步修正及重新验证另列，原失败记录保留。

## 同步修正参考

64组警告全部来自原参考 CTA64/KMMA4 内核的最终输出重排：64个 scalar shared store 后，
16个跨 lane shared load 之间缺少 warp 发布同步。actual PC 和原头逐条对应，C6 没有该
竞争警告。现有 CTA barrier 已在写入前，不是这个写→读缺口的替代。

仅在该 warp 自己的 BF16 输出重排写循环后补一处 `__syncwarp()`，见
[精确补丁](reference-sync.patch)。逆删该块后整个头与原归档逐字相同；K/V loader、
算术、tiling、planner、merge均未改。没有直接套入旧 BF16 参考的另一处 CTA barrier。
有效 header SHA256 为 `89b891802bf99bd2989c603d0c3346e0077a832d0c04199f3201d9b3e11023f8`。

独立 namespace 为 `g04_syncfixed_same_fp8_flashinfer`；小桥重编19.626秒，实际 archive
SHA256 为 `903576718caa737f73eede52f634ac7c154b85bdf44e914a6ec8e12b7e6809d3`。
ptxas 在该处实际生成 NOP，没有 WARPSYNC 助记符，不能把源码指令名当机器码事实；
修复充分性由实际 racecheck 验证，而不是为了某个助记符再加同步。

prefill＋tail 的数值、readonly、memcheck、synccheck全部通过，racecheck45.601秒后为
0 errors/0 warnings。window也通过数值及三种 sanitizer。主 prefill 最大绝对误差
0.00109081300601、window为0.000919902977887；各 case 两边有限 fixture 输出逐 word 一致。

最终普通计时 binary 为
`f65b9b27ce5842c4c303b2888a9df3a634ca6358950a10e799385c18a8ef60a9`。
namespace、SCHEDULE、provider 标签均为同步修正参考，未将其暗称原样上游性能。

## 实际同格式普通调用结果

实际设备为 NVIDIA GeForce RTX 4070（SM89），运行于 WSL2。两 case 各只有一次
固定普通采样：两分配顺序×两调用顺序×32blocks×128pairs，
各16384pairs，共32768pairs，全部长尾保留，无追加、裁剪或漂移扣除。

| case | 同步修正上游 FP8 µs | Hyle C6 FP8 µs | Hyle 均值变化 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| prefill | 84.989458 | 78.003444 | −8.2199% | [−8.7286%, −7.7111%] |
| window | 39.540493 | 37.048305 | −6.3029% | [−7.2453%, −5.3604%] |

prefill 四矩阵变化为 −8.0972/−8.4585/−9.1208/−7.2265%；window 为
−7.1343/−6.6617/−4.5657/−6.8814%。更快的 block 分别为128/128、116/128。
个别矩阵的 p99/max 仍有 Hyle 更差项，不宣称所有尾指标均改善。
区间仅描述本进程分块，不是跨进程、跨设备保证。没有与旧的 BF16 影子缓存对照拼接。

主 prefill 两边实际 tile64/padded40/split1、BF16 partial/F32局部累积；Hyle显式开启
`allow_bf16_partials`，不能将这个结果推广为默认 F32 partial 的性能。window 两边实际
split0、无 partial，不存在此前担心的不同 partial 类型混比较。

prefill required float workspace 为上游21,053,440B、Hyle10,567,680B。但本轮两边刻意
统一 cap64MiB，每 arm 两 slot 的实际 resource used/reserved 增量均为137,915,392B。
不能把本次 required 下降说成本次实际预留显存减半；C1 的紧 cap 内存测量另行保留。

## 未通过的辅助 case

D64/128 非因果辅助 holdout 在同步修正参考侧先失败 double oracle 的原8e-3阈值，
原始日志保留，未放宽阈值、未改数据，也没有对这个 case 做性能计时。此轮该 case
尚无公平性能结论，不能用另外两个 case 的成功代签。没有据此宣称全部 G04/G05 或
完整模型、全部精度/硬件域已经通过。

## 原件与复算

[raw.zip](raw.zip) 保存231个文件，包括消费者与参考桥源码、编译/链接/运行脚本、
完整命令与源码/二进制身份、原失败、八项实际验证、两次完整普通计时、独立审计与
统计复算脚本。归档为1,764,634B，SHA256：
`8598a3f8484d0e031f351d6f64a44dca14ec6dc9b24e2a271d4a8f3db8d95e67`。
[清单](archive.json) 逐项记录字节数与 SHA256；大型 ELF/object/archive 不重复入库，
其实际本机路径与 SHA256 单列为 external artifacts，不伪称已归档。

zip 内的 `independent/same-fp8-runtime-review.json` 逐次核对验证与普通计时身份，
`independent/same-fp8-{prefill,window}-analysis.json` 与 `*-all-matrices.csv` 保存全部
样本复算及四矩阵明细。归档已重开并逐项核对字节数与 SHA256。
