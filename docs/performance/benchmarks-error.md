# error 实测铁证

> 本页是 [performance 总账](./README.md) 的 **error 模块**详节（`Status` / `Result`）。
> 模块归属：`include/hyle/error/`、`bench/error/status_bench.cc`（核心 carrier）和
> `bench/error/normal_bench.cc`（正常用户 workflow）。hash / SwissTable 不在本页。

## 当前冻结证书（2026-07-24，三平台 n=15）

> 产品 commit `3539169`；Windows MSVC 19.50、WSL GCC 14.2、Apple M4 AppleClang 21，
> Release，Abseil 对照开启。方法为 `scripts/bench-stat.py -n 15`，同一进程内 pair 以
> `[sign]` 为判定。下表顺序统一为 **Win / Linux / M4**，绝对 ns 不跨机器比较。

核心与正常路径必须是两个 executable：把 workflow 追加进核心 TU 曾改变 MSVC
whole-TU inline budget，使完全相同的 Result error helper 从约 10ns 假退化到 13ns。
物理拆分后，被测产品 codegen 不再随 benchmark 体积变化。

### 核心 carrier

| 操作 | HYLE ns/op（Win/Lin/M4） | 对照 ns/op（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| `Status::Ok` 构造+检查 | 0.554 / 0.050 / 0.226 | absl 0.558 / 0.059 / 0.226 | 三平台 TIE |
| Status OK 拷贝 | 0.560 / 0.277 / 0.241 | absl 0.562 / 0.275 / 0.264 | Win/Lin TIE；M4 **WIN 1.094x** |
| code-only 错误 | 0.548 / 0.058 / 0.224 | absl 1.698 / 1.028 / 1.202 | **WIN 3.135 / 17.294 / 5.357x** |
| 带消息错误构造 | 28.052 / 9.038 / 15.424 | absl 53.960 / 21.928 / 29.005 | **WIN 1.924 / 2.409 / 1.926x** |
| heap Status 拷贝 | 10.322 / 8.974 / 3.181 | absl 10.805 / 9.380 / 4.166 | **WIN 1.043 / 1.049 / 1.309x** |
| 一层语义 context | 35.392 / 21.209 / 16.247 | StatusBuilder 298.042 / 103.886 / 114.979 | **WIN 8.478 / 4.887 / 7.031x** |
| `WithContext(OK)` | 0.742 / 0.277 / 0.480 | 无同形入口 | header-inline no-op，零分配 |
| Result OK 传播 4 层（裸） | 0.758 / 0.184 / 1.073 | StatusOr 1.096 / 0.184 / 1.073 | Win **WIN 1.477x**；Lin/M4 TIE |
| Result OK 解包并消费 | 1.440 / 0.336 / 1.589 | std 1.435 / 0.335 / 1.589 | 三平台 TIE |
| Result ERROR 上抛 3 层 | 9.847 / 10.260 / 3.206 | StatusOr 22.326 / 22.335 / 6.746 | **WIN 2.271 / 2.171 / 2.104x** |

`std::expected<int,std::error_code>` 只作为平凡错误下界；它不能携带消息，因此 ERROR
传播不与 Status/StatusOr 宣称输赢。

### 正常用户路径

| 操作 | HYLE ns/op（Win/Lin/M4） | 对照 ns/op（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| `Result<void>` OK 传播 | 2.070 / 0.291 / 0.494 | 正确手写 2.111 / 0.684 / 0.491 | Win/Lin WIN；M4 TIE |
| `Result<void>` ERROR 传播 | 9.895 / 8.879 / 3.301 | 正确 move 手写 10.172 / 9.243 / 3.299 | Win/M4 TIE；Lin **WIN 1.047x** |
| caller-buffer 人读格式化 | 10.727 / 9.989 / 8.028 | 无同形 Abseil 入口 | C/FFI/logging 基础路径 |
| C++ owning `to_string` | 34.109 / 20.028 / 22.626 | absl 42.621 / 25.928 / 27.833 | **WIN 1.247 / 1.302 / 1.212x** |
| 首错聚合 `Update` | 9.425 / 8.891 / 3.181 | absl 10.771 / 9.248 / 4.249 | **WIN 1.143 / 1.045 / 1.336x** |

空间不变：`Status=8B`，`Result<int>/Result<void>=16B`；`absl::Status=8B`，
`absl::StatusOr<int>` 在 Windows 为 24B、Linux/M4 为 16B。

## 本轮真实回退与修复

初次在 `768e2f2` 上复证时，旧平台翻转稳定复现：Linux heap copy、`Update` 与
`Result<void>` ERROR 约慢 9%；M4 OK copy 慢 28%。源码/对手审查发现 Abseil 将 retain
原子加法内联，而 HYLE 的 retain/release 都跨 shared-library 调用；HYLE 的 heap tag 判断
在 AArch64 还需要两个相关条件。

`3539169` 只做表示层优化：内联合法 heap rep 的 relaxed retain，并把合法 tagged word 的
heap 判定收成一个范围比较。API、8B 布局、C ABI、分配与最后释放路径均不变。A/B 后：

- Linux heap copy `0.910x LOSE → 1.049x WIN`，`Update`/Result ERROR 一并转 WIN；
- M4 OK copy `0.719x LOSE → 1.094x WIN`；
- Windows 保持全部 WIN/TIE。

因此当前功能面上的同形核心与正常路径三平台 **没有已确认 LOSE**。这不是“所有可能的
错误系统工作负载都已穷尽”；payload、序列化、backtrace 等未进入产品面的能力也不靠本证书
虚增 benchmark。

## 复跑命令

```bash
# 需 HYLE_BUILD_BENCH=ON、HYLE_BENCH_ABSL=ON，并指向固定 Abseil checkout。
cmake --build <build> --target hyle_error_bench hyle_error_normal_bench

scripts/bench-stat.py <build>/bin/hyle_error_bench -n 15 \
  --pair "absl::OkStatus" "HYLE Status::Ok" \
  --pair 'absl::InvalidArgumentError("")' "HYLE Status(code) inline" \
  --pair "absl::NotFoundError(msg)" "HYLE Status::NotFound(msg)" \
  --pair "absl::Status copy (error)" "HYLE Status copy (error)" \
  --pair "absl::Status copy (OK)" "HYLE Status copy (OK)" \
  --pair "std::expected (unwrap + use)" "HYLE Result (unwrap + use)" \
  --pair "absl::StatusBuilder SetPrepend" "HYLE Status::WithContext" \
  --pair "absl::StatusOr (manual check)" "HYLE Result + ASSIGN_OR_RETURN"

scripts/bench-stat.py <build>/bin/hyle_error_normal_bench -n 15 \
  --pair "manual Result<void> checks (OK)" "HYLE RETURN_IF_ERROR(Result<void>) (OK)" \
  --pair "manual Result<void> checks (ERROR)" "HYLE RETURN_IF_ERROR(Result<void>) (ERROR)" \
  --pair "absl::Status::ToString owning" "HYLE to_string owning" \
  --pair "absl::Status::Update" "HYLE Status::Update"
```
