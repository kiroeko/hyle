# Formal 性能证据保留层 v1

> **PENDING / Formal blocker。** 统一性能仪器当前已能对仍存在的文件做深重放；本任务解决长期
> 保留与证据真实性。它不阻塞普通模块的 engineering 收口，但完成前不得发布 Formal 性能证书。

## 问题

当前 manifest 引用 live build tree、compiler/linker 和产品 binary。后续 rebuild 会让 hash 校验
fail closed，因此不会产生假 PASS，但旧 bundle 也无法长期复核。本地 JSON/hash 还能证明记录彼此
一致，不能证明一个有权重写全部文件的人没有合成整套证据。

## 冻结范围

实现一个受信、内容寻址、不可变的 evidence bundle：

- 原子归档 contract lock、全部 raw stages、raw-log index、observations、controls、summary、host probe、
  validator logs 与 product artifacts；
- 归档实际 CMakeCache、CMake compiler/system records、链接命令与依赖闭包；
- compiler/linker 本体仅在许可允许时进入内部 CAS，否则绑定受信的 toolchain provenance/安装映像；
- manifest 只使用 bundle 内相对路径与内容 digest，根清单在一次稳定快照后签名；
- 归档与签名消费同一个已打开的稳定快照，certificate 只引用该快照 digest；
- 定义 key custody、访问控制、retention period、导出/删除与许可证审计。

## 非目标

- 不把普通 engineering probe 变成远程 CI 服务；
- 不声称软件签名等于可信硬件测量；
- 不归档无许可的第三方 binary；
- 不改变统计、margin、停止规则或产品模块 API。

## 验收

1. bundle 离开原 build tree 后仍可完整 `verify`；删除/重建原 build 不影响结果；
2. 任一归档文件缺失、替换、重复或 root signature 不匹配均 fail closed；
3. n=15→n=31 两阶段和 child nonce 可从 bundle 独立重放；
4. Formal `qualify-platform --formal` 与 `certify` 只有在 retention/signing policy PASS 时开放；
5. Windows/macOS 两台独立登记机器的同 candidate bundle 能组合，live-path manifest 不能组合；
6. 文档明确真实性、许可和保留期边界，并有删除原 build 后的回归测试。
