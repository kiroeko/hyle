# CPU ATen：存储与索引接线资格

SI1–4 **CLOSED**，不可变产品候选 `91ad443bbaa20d51535d0bfcae145ba46b07b4a4`。
[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-storage-index-admission.md)；
Windows 是产品真相，Linux/Windows 静态 source tree 均仅单向同步，1332 个输入 hash 验证一致。

## 已实现的六族与普通入口

Select、Gather、ScatterAssign、Fill、Copy、Concat 使用已锁 LibTorch 2.8.0 CPU 原版程序，
全部15种 plain scalar 通过 Byte carrier 保留原始位；不把宽整数/FP8存储误当成数学支持。
public rank<=8、exact shape、稠密输出，原 Byte rank 可到9；广播条件非零判真，合法 odd-byte
stride、任意轴与 variadic Concat、scalar/多维 I32/I64 index、重复读与最后逻辑写入均已验。
Assign 原 stable sort/mask/select 后只向唯一 index 写入，未执行原 duplicate index_copy UB。
ScalarValue 仅冷构造一个 scalar bytes；无新增 payload 数学或 HYLE copy kernel。

普通 `compute::cpu::create_context()` 与 `graph::cpu::create_context()` 自动绑定。既有 mature
region/leaf 顺序保留，ATen 固定在 oneDNN/ggml 后、Native 前；准备时匹配，执行不搜索/计时。
原 expert header `providers/native.h` 提供独立 `aten_cpu::provider()` identity，不覆盖 Native
数学 factory。普通consumer只链接 hyle::compute/graph，不注册或排序Provider。

## 实际验证

| 环境 | 受影响测试 | 相同普通消费者的 C++ 交付 |
|---|---|---|
| Windows | Provider114/114；CPU12通过/1跳过；Graph15/15；公开头1/1 | source shared/static、installed shared/static 全通过 |
| Linux GCC14 | Provider114/114；CPU12通过/1跳过；Graph15/15；公开头1/1 | source shared/static、installed shared/static 全通过 |

每个普通 Direct 与 op→Copy Graph 程序变化输入重复执行、真实 wait、验证最终 bytes。
具名 UInt64/I64 Gather、Assign、宽Fill实际绑定ATen；Select/Copy合法域绑定已接oneDNN，
Concat绑定ATen。不接受 Native fallback 冒充新成熟接线；Graph 资格是可组合 leaf 与下游，
没有宣称新增全 ATen region。

新增8项批量测试含全scalar精确bits、负零/NaN payload/宽整数、rank0/rank8、empty、33个Concat
部件、各axis/两index宽度/三index rank。late invalid index在全部输出写入之前失败，输出poison
保留；failed invocation必须冷重建，重建后恢复。Copy alias拒绝；仅Scatter exact dense input0
alias通过。tracked冷prepare/invocation分配逐点OOM恢复、资源归零、close拒绝晚到工作；独立
两个invocation/queue实际两线程并发12轮结果隔离。原库物理OOM和非coherent映射故障未注入，
其异常翻译/flush/close传播做源码检查，不虚报成故障注入PASS。

旧 `D5RealQwenProjectionUsesExactProfileAndReplays` 缺fixture，两平台跳过不计PASS。本卡不改变
该模型画像；六组普通consumer与对应语义测试均实际运行。GPU、完整模型加载和新性能比较未运行。
Windows另外将当前ATen/CPU/Graph三个TU以关闭feature编译，验证invalid factory被两普通assembler
跳过；复用已安装依赖，不宣称重新构建完整SDK-OFF产品或清除Torch DLL依赖。公开头两平台自含。
Windows执行DLL均通过既有helper放在exe邻目录，不改变系统PATH或弹窗控制治理。

## 先行失败与有限纠正

初始Provider112/114：Copy误声明alias违反公共trait，产品仅收紧到Scatter；恢复测试错误复用
failed槽，改为拒绝旧槽并重建。一次产品纠正后最终114/114。普通consumer错误强制六族都选ATen，
与已有oneDNN重叠；只修正qualification identity假设，精确结果、宽索引/Fill绑定要求保留。
关闭feature consumer错误假设oneDNN不存在，只改成无ATen绑定且保留既有成熟实现。
最终候选所有门的最新命令通过；全部初始失败日志/命令、原始XML保留，未抹掉历史。

原 API diagnostic 的454断言仍由[原manifest](original-manifest.json)绑定；只证明原程序可运行。
原SiLU/GELU的−Inf、Sigmoid尾部与amin signed-zero差异继续具名DEFER，不计本批math支持。

## 成本、终审和剩余

Select/Fill/Copy/Concat O(payload)，Gather包含O(indices)检查与必要packing，Assign包含
O(N log N)稳定排序、winning update materialization、base copy。cold HYLE metadata与native
condition/index slots复用，native Tensor metadata/sort/scratch仍是managed replay；同步CPU执行
阻塞host，原库私有线程不受ggml的native_threads控制，不能承诺capture、零热分配或精确内存上界。
硬预算/Prepared-only通过实际属性拒绝不合格候选；未报告的内存/copy/dispatch保持unknown。
这符合声明成本的P0资格，不产生性能WIN，不启动冻结自研比较或优化。

终审reviewer root，2026-10-06，独立于实现的一遍SI1–4源码/资格证据检查；alias/失败槽/宽math
误准入/重复写UB/映射顺序/冷热资源/普通identity/三交付/关闭feature边界均按上述处置收口。
七门范围/架构/正确性/交付/用法/红队/P0通过；取消/截止的kernel内部抢占不属于同步原库新承诺，
不制造该能力。只有本批可复现语义、资源、组合或交付反例/相关依赖升级可重开。

剩余CPU数值成熟候选及自研逐域裁决仍未完成；bounded形状、其它alias、严格预算与本卡外math
不能继承本批资格。自研优化/性能清退冻结，SIMD基础保留，其它路线保持暂停。
统一Compute + Graph候选实现LOC **61,764 / 249文件**，不含测试/vendor。
本批消费者临时目录清理 **913,563,267 bytes（约914 MB）**，保留producer/SDK/cache；
原probe的DLL硬链接不计成实际回收267MB。

[收据](receipt.json)与[工件](qualification.zip)包含命令、全部失败、XML、同一consumer/stub、
输入hash、selector selected/skipped、驱动、cleanup与候选LOC；不复制SDK或build大目录。
工件SHA256 `c35876a4e511b8027197db3715498467ca30c9763e3f3fae3af101c38068060e`。本卡关闭仅文档/状态，无重复构建/P1。
