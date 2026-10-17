# CPU resident SDPA 比较

原始实验身份：产品 `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，适配器
`901d926f4f9b14037d4586d0f3f170a199746516`，CPU 原核 `ffa4e8b80930`。
历史 contract v2 的 SHA 为 `d766ab8f24fbc7f0ea8ad0fffa66949e2f04dba675188a8f014a75b7e62606f2`；
旧 JSON、停止记录和原始结果保留，不再控制当前开发与采样。

形状为 BF16 BHSD、B1/GQA8/2、容量尾 13：Q1/128、KV1024/D256 causal，以及
Q17/KV257/D64 noncausal。同 ready 输入、readonly cache、两个 Invocation、一个 native thread，
经 CPUQueue 提交、checked Completion.wait 后返回 caller output。

产品与独立参考都使用 PreparedOperation/typed kernel 生命周期。参考直接构造原 flash graph，
不调用产品 supports/lower，也不写新数学。热计时包含 query cast、BF16 输出转换与 native math；
冷 metadata、F16 mask 和 cplan workspace 分列。重放不重建 metadata/graph/plan。

复杂度为原 flash `O(H×Q×KV×D)`，不展开 KV 到 Q/head 维度。query/payload 为 `O(QHD)`，
causal mask 为 `O(Q×KV)`，workspace 依 cplan；native/private pool 上界未知，不能报告成零或总预算保证。

本目录是独立 CMake 入口。用对应静态 CPU build 构建 `g01_cpu_resident_bench` 与按需
`g01_cpu_resident_bench_placement`，直接执行；`HYLE_BENCH_VERIFY_ONLY=1` 做数值与重放检查，
`HYLE_BENCH_AA_ROUTE=native|selected` 做同路径 A/A。实际 affinity、线程数、执行/分配顺序与
输入配置随输出保留，使用[性能比较方法](../../../docs/developer/performance/README.md)分析重复运行。

原始记录已验证两 placement 的 double oracle、finite、readonly 与双 slots，未给出最终正式计时。
这只说明当时的检查范围；当前数字需要从本次实际产物取得。
