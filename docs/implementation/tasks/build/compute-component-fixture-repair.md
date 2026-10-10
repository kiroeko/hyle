# C3 独立工具修正：组件合同的最小源码夹具

<!-- MODULE REVIEW: AUDITED -->

- tool owner：`cmake/HyleCiComponentContract.cmake`；baseline `3067016ca`。
- scope：只迁移两个最小源码夹具的文件依赖，不改产品、组件图、路线规则或公共治理核心。
- 真实反例：当前组件合同退出1，Core selected-audit child 的合成源码根缺少
  `cmake/HyleComputeRoutes.cmake`，实际 manifest 在原断言之前 include 失败。
- 原失败日志/命令/exit 已保留于
  [component-contract evidence](../../../../evidence/correctness/compute-unified-c3/component-contract/)。

## 一次审计与固定 FIX

`HyleComponents.cmake:690–691` 从 `HYLE_SOURCE_DIR` 取得路线解析器；这两个夹具故意把
该变量设为最小源码根，用于验证未选 source 不审计、选中 source 的非法直接边必须拒绝。
源码扫描确认只有 selected-audit/selected-invalid 两个 child 覆盖实际源码根；其它 child
仍引用真实根。本轮只把同一真实解析文件复制到两夹具，不复制或改写解析实现。

固定 FIX：在两个 child configure 前各复制一次 `HyleComputeRoutes.cmake`。
正例必须继续成功，反例必须继续以原 undeclared-edge 原因失败；不得删除/放宽断言。

P0：测试准备的一次文件复制，无产品运行期开销。没有产品 owner reopen 或新 component。
预算：一遍上述审计、一个夹具修正、一次完整 component contract；若另有失败，先保留具体
反例并定界，不启动全库数学、不重置 C3 已通过证据、不重建 SDK、不跑 P1。
适用检查：该原合同、docs/closure/diff、selector 对工具变更的记录；原 native batch 继续。

当前已完成 SCOPED/审计，有限 FIX 已固定；实现与修正后验收尚未签。

## 后续具名反例与再定界

第一 FIX 两夹具已实际通过原正/反断言，工具候选 `788f9e80`；完整合同后段第一次因
本机非系统路径的 Vulkan SDK 未传入而失败。传入已使用的标准 `VULKAN_SDK` 后只续跑
未完成后段；all roots、IO、嵌入、依赖 cache、vendor/package registry 均继续通过。
该次在嵌套 target-platform 合同失败，日志保留，未把任何后段未运行项写 PASS。

实际新范围：`HyleCiTargetPlatformContract.cmake` 的 unsupported-fcontext metadata child
仍 `LANGUAGES NONE`，未建立 target identity；普通完整 CPU 路线解析因此先以 `Windows/`
拒绝，尚未到原 Sched 依赖断言。再审计并冻结唯一 FIX2：该 child 以真实 CXX 初始化，
调用既有 `HyleTargetPlatform.cmake` 得到真实 ABI/目标身份，再设置模拟的 fcontext=false。
不写死 x86、不删除 CPU admission、不放宽原 Sched/32-bit/目标身份断言，不改任何产品实现。

预算新增仅为一个上述具名 fixture 纠正、一次既有 target-platform 合同以及原 master 在其后
尚未运行的尾段。已通过前段和 GPU SDK 不重跑，公共平台/selector/perf 核心仍不改。
本工具保持 AUDITED，范围仅追加这一个 fixture；未授权新产品能力或全库重审。
