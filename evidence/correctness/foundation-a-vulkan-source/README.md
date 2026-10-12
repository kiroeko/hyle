# Foundation A Vulkan：首批原生与混合 source 资格

活动卡：[Graph 对称性](../../../docs/implementation/tasks/graph/foundation-a-route-symmetry.md)、
[Vulkan 冷控制接缝](../../../docs/implementation/tasks/compute/foundation-a-vulkan-control-seam.md)。
本目录只签 Windows source 初批旅程；Apple、installed static/shared、独立头、迁址、
sanitizer、effect 端口和 red team 尚未由本证据裁决。

实际调用 [route_symmetry_vulkan.cc](../../../tests/graph/route_symmetry_vulkan.cc)。
Windows MSVC19.50 / Release / shared shipping libraries / HYLE_TESTING=OFF，原 native Vulkan
Context 默认拒绝 software device；复用原独立 w-vulkan 和 w-cpu-vulkan source 树，库 j4/test1。
实际 driver 身份的最终候选 receipt 仍需随资格汇总取证，不用软件 driver 代签硬件。

原生旅程 PASS：B=1,4 两个冷 artifact、B=2 明确表外、同 artifact 两个独立 invocation 的
输出与旧结果保活；仅作控制前驱的 checked Gather 被保留，非法页外 index 导致 OutOfRange，
合法前驱后继算术正确。底层沿原 ggml/IREE/commands，不新增数学 kernel。

混合旅程 PASS：原生旅程同上，加 CPU→Vulkan 两个 exact shapes 的数学、旧 lease 禁止 reset；
GPU control-only Gather→CPU 后继保持两个 fragment，host_wait=kReject 拒绝，原 managed
等待允许时失败正常传播。无数据 consumer 的前驱没有被剪掉。

原 native build 138.203 秒、journey0.609 秒。hybrid 首轮 build156.406 秒 FAIL：新 fixture
误用了不存在的 compute::cpu::Context factory；修成原 graph::cpu::create_context。
同时去除产品 reverse pass 的 unused operation warning，不改变算法。原失败完整保留。
同树增量 build7.766 秒、journey0.500 秒 PASS。hybrid 累计 build164.172 秒。
没有因新目录重置原 gate 的预算；没有回放全部 Graph 算子或 F1–F3 验证。

job receipts 保存各编译输入 hash、命令/进程、j4 与耗时。实施快照不是 CANDIDATE，
本轮后来仅补公共头合同说明/nonthrowing artifact handle 编译断言和 fixture 错误打印；
最后 hybrid 已实际实例化并运行 native 与 hybrid finite families。其他待验格仍 UNPROVEN。
