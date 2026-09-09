# HYLE 维护脚本

本页是当前入口索引，不复制工具的完整 `--help`、历史模块命令或一次性调查。规则分别见
[维护者文档](../docs/developer/README.md)、[构建架构](../docs/architecture/build-system.md)和
[性能仪器](../docs/developer/performance/instrument-v1.md)。

## 日常入口

| 任务 | 入口 | 合同 |
|---|---|---|
| 按 Git 影响选择验证 | `validation-select.py plan/run` | 文档/测试不触发产品全量门；产品 owner 必须先处于 `AUDITED`；P1 只接受显式冻结卡 |
| 校验/投影模块状态 | `check-module-closure.py` | `module-status.json` 是唯一状态；registry 改动必须传 `--previous-ref` |
| 校验文档 | `check-docs.py` | 四类目录、相对链接、退役 FFI 和从 `docs/README.md` 可达性 |
| 统计代码 | `loc-report.py` | 全库、自实现不含测试、单 factor 或 registry 状态集合；支持 immutable commit |
| Unix 工具链入口 | `hyle-unix.sh` | 选择受支持 GCC/Clang，拒绝半配置环境，不安装系统工具 |
| Windows 工具链入口 | `hyle-windows.ps1` | 选择已安装 VS/SDK，支持独占 `-BinaryDir`、factor 与产品 smoke |

最常用的治理命令：

```text
python scripts/validation-select.py plan --base <baseline> --candidate WORKTREE --output .scratch/validation/plan.json
python scripts/check-module-closure.py --previous-ref <registry-before>
python scripts/check-module-closure.py --write-status
python scripts/check-docs.py
python scripts/loc-report.py --registry-state CLOSED --commit HEAD --self-check
```

只有产品候选提交后才执行 selector 的 native gates；精确 context、receipt cache 和 P1 card 格式见
[按变更影响验证](../docs/developer/review/change-impact-validation.md)。

## 构建与交付助手

- `header-self-contained.py`：逐头自包含/安装头合同；
- `core-codegen-audit.py`、`core-symbol-visibility-audit.py`、
  `container-intrusive-codegen-audit.py`、`time-codegen-audit.py`：各 owner 的 P0 结构门；
- `simd-exact-artifact-check.py`、`simd-provider-codegen-audit.py`：SIMD 固定 ISA artifact/codegen 门；
- `rvv-qemu-test.sh`、`sve-qemu-test.sh`、`simd-scalable-exact-qemu-test.sh`：明确的 scalable ISA
  交叉执行门；
- `fetch-bench-vendors.{sh,ps1}`：取得锁定的外部 SIMD peer；
- `setup-bench-libs.sh`：按需取得开发期强 peer，不是产品依赖锁或 freshness 证明；
- `check-no-vendor.sh`：防止 vendor 类型/include 越过 SIMD 公开边界。

源码依赖由 `cmake/HyleDependencyLock.json` 唯一登记，默认进入跨 build 的
`HYLE_DEP_SOURCE_CACHE`；不要用 benchmark fetcher 替代产品依赖解析。完整准备方式见
[依赖](../docs/developer/setup/dependencies.md)和[离线构建](../docs/developer/setup/offline.md)。

## 性能入口

`hyle-perf.py` 与 `hyle_perf/` 是唯一当前性能证据核心：

```text
python scripts/hyle-perf.py validate --contract <contract.json>
python scripts/hyle-perf.py probe ...
python scripts/hyle-perf.py qualify-platform ...
python scripts/hyle-perf.py verify --manifest <manifest.json>
python scripts/hyle-perf.py certify ...
```

具体参数以子命令 `--help` 为准。合同在 `bench/contracts/`，机器身份在 `perf-machines.json`，schema 在
`perf-contract-schema-v1.json`；结果必须遵守
[性能证据标准](../docs/developer/performance/benchmark-evidence-standard.md)。

`bench-stat.py` 和 `op-parity-report.py` 是仍被当前合同或 SIMD 诊断使用的窄适配器，
不能单独签署模块。`validate-*-bench.py` 由具名 contract 调用；新增 workload 优先增加 contract/plugin，
不要复制一个新的模块 runner。`mem-footprint-stat.py` 仍服务 RSS 隔离实验，但不构成当前证书。

## SIMD 生成与路由

- `simd-op-coverage.py`：从公开面生成/校验全操作清单、provider 发现和 terminal route；
- `simd-op-routing.json`：逐操作审查账本；
- `simd-family-routing.json`：whole-buffer family 路由；
- `simd-op-manifest.json`、`simd-scalable-op-manifest.json`：生成物输入；
- `simd-route-qualification.py`、`simd-scalable-coverage.py`：资格与 scalable 覆盖检查。

这些大 JSON 是机器合同，不是人读项目状态。不得手工改 generated route；使用具名脚本、锁定 provider
和审查 receipt 更新。设计与用户形状见
[SIMD 架构](../docs/architecture/simd-design.md)和[SIMD 用户指南](../docs/user/simd-usage.md)。

## 保留边界

脚本只有在当前构建、测试、合同、文档或 terminal evidence 中存在 owner 时才留在主分支。被统一仪器
替代的 `run-*-bench-evidence.py`、孤立 `*-stat.py` 和相应 parser test 不保留作兼容层；需要复现实验时
从 Git 取回当时版本。删除旧脚本不得改写原始 evidence，也不得把历史输出冒充当前 qualification。
