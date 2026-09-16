# R4 C6：CUDA 私有 context bridge

产品提交 `42ae3005899cbd6117494b2a609afae6133eae6b`。
已有 ScopedPrimaryContext 的声明/导出移到小型私有头，新增只读 primary-context borrow；
原 context push/pop 算法不变。供后续同构建 Provider 复用。

W1-WIN、MSVC 19.50、CUDA 13.3、RTX 4070；shared、Werror、Ninja 并行度 2，
只选 compute_backend_cuda factor。三个适用 source 测试全部 PASS：公共头、backend 与隔离晚到失败。
实际 HYLE/CUDA/MSVC DLL 已复制到 exe 同目录，非交互 CTest，单测试 timeout 120 秒。

`windows-shared.zip` 保留 configure/build/test、DLL staging manifest、selector 原始计划和 LOC；
`source-hashes.json` 锁定三个产品文件；`sha256.json` 锁定证据文件。
selector 只选 docs、closure、configure 和 CUDA backend，不选 CPU/oneDNN/P1。

这仅证明 C6 的窄改动回归；R4 cuBLASLt/Marlin/HYLE Provider 的实际消费、完整 Direct/Graph、
source/static/shared 交付和性能尚未完成，不由这个 backend-only 结果代签。
