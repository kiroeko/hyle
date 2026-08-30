# SIMD-16：consumer 迁移与旧机制清算

- ticket state：`DONE`
- kind：product migration / deletion
- budget：1 工程日；P1 60 分钟
- depends on：SIMD-15

迁移 ordinary families、source expert、Container 与 Ordering 的真实引用到已签 route。只有某旧定义的全部
ledger 项都有终态、获胜实现已先提取、正确性覆盖已有新 owner、产品/测试/bench/CMake 引用归零时，才可
删除该定义或 backend。每批删除后运行受影响 correctness、consumer sentinel 和 route completeness check；
禁止用安装过滤代替物理 ownership 清算。

## 结果

- ordinary row/dtype/math 与 source fixed/fat seam 使用同一 compile-time provider route；未知 CPU 的普通
  入口仍只在 whole-buffer 边界经过既有 cached target call，没有新增 provider runtime dispatch。
- VQSort adaptation 已直接使用 locked Highway native values；Container 的 bitset/string-search consumers
  也迁移到中央 Highway product dependency，不再依赖 HYLE register façade。Container 仍不依赖
  `hyle::simd` product target，组件图没有形成反向环。
- `include/hyle/simd/vec.h`、`native.h`、`native/*`、七套 `backend/*` 及其专用测试/benchmark 已物理删除；
  全仓产品、测试、bench 与 CMake 对这些路径的引用为零。冻结 ledger 继续保存 baseline source/site
  provenance，因此删除源码不删除审计事实。
- 迁移期 ownership correction 将 10 个 VQSort-only composition/facade 从 `KEEP_HYLE` 改为
  `ROUTE_HIGHWAY`，并把 `Stream` 的 HYLE 例外限制为 AppleClang 21 NEON；证据见
  `evidence/simd-16-migration-route-corrections.json`。终态为 387 operation、422 route scope：366
  `ROUTE_HIGHWAY`、34 `KEEP_HYLE`、22 `REMOVE_REDUNDANT`、零 `PENDING`。
- `python scripts/simd-op-coverage.py --check-ledger --require-route-decisions
  --check-provider-manifest --check-provider-routes` 为 `PASS`。旧 source-coverage `--check` 在物理清算后
  按设计不可用，不能误作终态命令。
- provider route/codegen/winner、ordinary SIMD 与 ordering 五个 test artifact 在 W1-WIN、W1-WSL、M1-MAC
  的当前 release 构建均为 5/5 `PASS`；最终不可变 candidate 的交付与 sanitizer 回放归 SIMD-17。

terminal verdict：`DONE`。所有删除均满足“ledger 终态、赢家先提取、新 owner correctness、引用归零”
四项门；SIMD-17 不再拥有新增产品能力或重写路由的权限。
