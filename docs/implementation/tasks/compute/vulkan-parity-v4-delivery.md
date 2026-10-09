# Vulkan V4：当前候选的平台与三交付

<!-- MODULE REVIEW: SCOPED -->

baseline `d04978bd`；协调owner为当前Vulkan/Graph，build/package是横向交付工作。
[父轮](vulkan-cpu-cuda-apple-parity-closure.md)全部需求保持。V3性能资格条件已获
[明确工程取舍](vulkan-v3-performance-engineering-decision.md#用户明确决定2026-10-09)，本包独立验证，不代签P1。
目前仅source/环境/旧交付接缝审计，不在CANDIDATE/BLOCKED owner下写产品。

有限结果：Windows与Air真实可用设备的source/static/shared与搬迁、DLL/dylib邻置、
内嵌SPIR-V/VMFB、许可证、公开头、offline、OFF不探测/下载其它路线；WSL实际GPU
缺席时只签build/header/拒绝行为，不能以software device冒充GPU数值资格。
两个普通入口执行已审固定模型/量化/状态/失败资源范围。只有新平台具体产品缺陷
才依法另定有限FIX和owner，旧CPU/CUDA/Apple及公共工具不顺手重开。

当前身份（2026-10-09）：Windows canonical/RTX4070资格继承本轮；Air Mac16,12/AppleM4/
16GiB、macOS27.0.1/26A434、AppleClang21.0.0、CMake4.3.4、剩余41GiB；实际MoltenVK1.4.1，
loader1.4.350、device API1.4.334、vendor106b/device1b000209。Air验证clone当前7365f0fb，
树干净；仅从Windows当前完整候选同步，不反向取产品。SDK工具/loader已在Homebrew。
WSL Ubuntu24.04/6.18.40.1/原ext4，尚未确认真实Vulkan GPU；旧llvmpipe不继承为本轮PASS。

预算：交付source审计20分钟，批量三交付每机一producer/两配置，build2/test1，重型串行。
复用锁定sourcecache与有效producer，精确清理本轮拥有者临时包，不删模型/共享cache。
每具名配置/fixture/ABI错误一次纠正；未适用与未运行分别记录，不以旧R6三交付代签。
selector根据真实API/package seam选择门；文档不重建。closure/docs/diff及统一LOC必需。
全部平台/交付及父轮性能取舍未结束前不CLOSED/不声称基础完成。
