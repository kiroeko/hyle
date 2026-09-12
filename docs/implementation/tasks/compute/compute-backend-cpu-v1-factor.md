# Compute Backend CPU v1：主机执行因子审计

## Closure card

- component / owner：`compute_backend_cpu` / Compute；round=`compute-backend-cpu-v1-factor`；
- baseline commit：`ec83514fe9d082698eb05018680cfaeda1822a5a`；
- trigger 与证据：当前 `compute` 同时拥有通用 runtime、CPU device/queue 和 HYLE SIMD/native kernels，导致
  后续 CUDA/Metal-only product 无法只取得 runtime，并阻碍 backend 与 provider 并行施工；`Device::cpu()`、
  全局单项 `enumerate_devices()` 和同步 queue 已提供可迁移的 CPU 边界；
- 正常 consumer 任务：普通 C++ 用户枚举/选择 host CPU，建立 context 和 ordered queue，取得 host
  `MemoryResourceRef`，执行 direct/prepared work，显式 close/drain；空 CPU completion 仍与异步 backend
  使用同一 runtime 合同；
- 主产品纵切：HYLE-native/oneDNN 的 transformer CPU prepared operation；正交纵切为不经过 Graph 的 typed
  host engineering kernel；
- scope：逐入口裁决并迁移 CPU device factory/enumeration/properties、backend context/queue、host resource 与
  CPU native interop；验证它只实现执行轴，不拥有 Operation schema 或 kernel；
- non-goals：任何算子/reference/SIMD kernel、oneDNN/ggml、GPU backend、线程池、Graph、Native FFI；
- 直接依赖 / 消费者：候选公开依赖 Compute runtime，私有复用 Platform/Mem；直接消费者为
  `compute_provider_native`、oneDNN、direct CPU Compute 和 Graph；
- 性能：`P0` 优先，CPU create/submit/close 不隐藏线程、allocation、syscall 或 provider search；若 runtime
  seam 比直接 caller-thread function 有无法由合同解释的 material 成本，候选后只跑一次同合同 `P1`；
- 验证：无设备/非法 ordinal、properties、resource identity、queue ordering、并发 queue、close/drain、
  context/queue/completion 销毁顺序、错误注入、逐头、三平台、sanitizers、source/static/shared、factor-only、
  P0/必要 P1 和红队；
- 预算 / stop：API 与 owner 最多两轮，FIX 各最多两轮；若通用 runtime 尚不能承载 backend state/resource
  lifetime，返回 Compute v4 而不是在 CPU factor 私造另一套 Context/Completion。

## SCOPED 待审计项

1. `Device::cpu()`、全局 enumeration/query、Context/Queue 当前实现分别 `MOVE/REPLACE/DELETE` 到哪里；
2. CPU backend 是否只需 caller-thread ordered queue，怎样使用通用异步 terminal 而不增加 heap/atomic；
3. `mem::system_resource()` 是公开默认 resource 还是由 typed CPU adapter 显式返回；
4. source/static/shared 与 Compute product façade 的唯一投影。

AUDITED 前不移动 CPU 产品代码或添加依赖。
