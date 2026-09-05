# Platform-thread v2：CPU feature consumer 迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只处理 Platform v4 删除旧 `CpuFeatures` 字段形状后，platform-thread 对 SVE/RVV/AMX
> 主机事实的直接消费迁移；不重开线程、affinity、Parker 或 ISA-state 机制。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform_thread` / Platform /
  `platform-thread-v2-cpu-feature-consumer` / `REOPEN`；
- baseline：`749e67a58bd87efcfb6b1b836dc7cfeaaf0fdfd0`；前轮：
  `platform-thread-v1-factor`；
- trigger：`PLATFORM-THREAD-CPU-FEATURE-CONSUMER-2026-09-05`；
- concrete fact：`src/platform_thread/thread.cc` 直接读取 `CpuFeatures::amx_tile/sve/rvv`。
  Platform v4 已冻结删除字段结构、改用无顺序 `CpuFeatureSet`，所以该 CLOSED factor 的依赖/API
  接入必然发生产品改动，命中“public API/dependency changes materially” reopen trigger；
- 初步范围：只把上述查询迁到 `CpuFeatureSet::has(CpuFeature)`，并回放 thread tests、Platform façade
  三交付和 Sched direct consumer；不改变公开 platform-thread API、layout、错误、同步或控制流；
- 初步性能级别：`P0`。三处缓存 set membership 均为 O(1)，不重跑 Parker P1；
- 非目标：新增 CPU facts、修改 AMX permission/SVE/RVV vector-length 机制、线程或 wait 重构、公开 API、
  Native FFI、通用仪器及 Formal certification。

只读 replay verdict 为 `PRODUCT_TRIGGER`。进入 `SCOPED/AUDITED` 前不修改 platform-thread 产品源码。
