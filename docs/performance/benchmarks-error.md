# error 实测铁证

> 本页是 [performance 总账](./README.md) 的 **error 模块**详节（`Status` / `Result`）。  
> 模块归属：`include/hyle/error/`、`bench/error/status_bench.cc`。  
> **2026-07-23 当前接口 Windows n=15；2026-07-20 旧核心形状三平台 n=15**。
> hash / SwissTable 不在本页——见 [mem / container 实测](./benchmarks-mem-container.md)（代码在 `container/`，非 util）。

## 当前接口（2026-07-23，Windows MSVC n=15）

> 候选工作树基于 `bfa90ac`，包含 `WithContext`、新传播宏、C ABI 收缩和
> CUDA/Vulkan/OS translator 重构；尚未提交，最终 clean commit 在提交轮补记。方法为
> `scripts/bench-stat.py -n 15`，同一进程内 pair 信任 `[sign]`。Windows MSVC 19.50
> Release，Abseil 开启。Linux/macOS 当前接口尚未复跑，因此本节不是三平台冻结证书。

| 操作 | HYLE ns/op | 对照 ns/op | 配对结论 |
|---|---:|---:|---|
| `Status::Ok` 构造+检查 | 0.379 | absl 0.378 | TIE |
| Status OK 拷贝 | 0.564 | absl 0.554 | **LOSE sign 0.984x**；绝对差 0.010 ns |
| code-only 错误 | 0.371 | absl 1.729 | **WIN 4.533x** |
| 带消息错误构造 | 27.636 | absl 53.992 | **WIN 1.941x** |
| heap Status 拷贝 | 9.375 | absl 10.718 | **WIN 1.146x** |
| 一层语义 context | 34.932 | absl StatusBuilder 317.594 | **WIN 9.073x** |
| `WithContext(OK)` | 0.760 | 无同形对照 | 头内联 no-op；零分配 |
| Result OK 传播 4 层（裸） | 0.752 | absl 0.982 / std 0.548 | **WIN absl 1.293x**；LOSE std（见下） |
| Result OK 解包并消费 | 1.430 | std 1.419 | TIE，ratio 0.991 |
| Result ERROR 上抛 3 层 | 9.975 | absl 22.044 | **WIN 2.193x** |

空间不变：`Status=8B`，`Result<int>/Result<void>=16B`，Windows
`absl::StatusOr<int>=24B`。

本轮性能判断：实际 SOTA 同形对手 Abseil 上，除 OK 拷贝的 0.010 ns sign 微输外全部
平或赢；新增 context 明确赢 9.07x。裸 OK 传播输 `std::expected<int,error_code>` 是
不同错误能力的下界比较；真实的“解包并消费”形状统计平局。当前唯一需要继续跨平台
观察的是 OK/heap 拷贝的 codegen 翻转，不需要为 benchmark 增加另一套产品类型。

## 旧核心形状（2026-07-20 三平台快照）

> 源：`bench/error/status_bench.cc` → `hyle_error_bench`。  
> 方法：`scripts/bench-stat.py -n 15`，信任 **[sign] 配对符号检验**（p&lt;0.01），不信单遍 relative。  
> 平台：Win MSVC 19.50 Release（`build-bench`）/ WSL GCC 14.2 Release（`/root/hyle/build-error-bench`）/ Apple M4 AppleClang 21 Release（`~/hyle/build-error-bench`）。  
> 门控：`-DHYLE_BENCH_ABSL=ON` 接 `absl::Status` / `absl::StatusOr`。  
> **两把标尺**：① 同款 SOTA = absl（tagged-pointer + gRPC 码表同源）；② `std::expected&lt;int,std::error_code&gt;` = 标准下界（E 平凡、**带不了消息**，只作基线，不可与带消息 ERROR 路径直接比）。

### 三平台中位数（ns/op，N=15）

| 操作 | HYLE Win/Lin/M4 | absl Win/Lin/M4 | std Win/Lin/M4 | 对 absl [sign] | 对 std [sign] |
|---|---|---|---|---|---|
| **Status::Ok 构造+ok()** | 0.551 / 0.053 / 0.226 | 0.552 / 0.053 / 0.226 | 0.561 / 0.183 / 1.073 | 三平台 **TIE** | Win TIE；Lin/M4 **WIN**（3.36x / 4.73x） |
| **Status 拷贝(OK)** | 0.592 / 0.237 / 0.362 | 0.561 / 0.276 / 0.256 | n/a | Win **LOSE 0.95x**；Lin **WIN 1.26x**；M4 **LOSE 0.72x** | n/a |
| **Status(code) inline 无消息** | 0.549 / 0.053 / 0.224 | 1.881 / 1.012 / 1.193 | 0.367 / 0.053 / 0.224 | 三平台 **WIN**（3.39x / 19.8x / 5.32x） | 三平台 **TIE** |
| **Status(msg) 带消息→堆** | 26.13 / 8.90 / 16.40 | 54.77 / 22.19 / 32.44 | n/a | 三平台 **WIN**（2.14x / 2.45x / 1.93x） | n/a |
| **Status 拷贝(heap,retain/release)** | 9.40 / 10.34 / 3.17 | 10.70 / 9.32 / 4.15 | n/a | Win **WIN 1.14x**；Lin **LOSE 0.90x**；M4 **WIN 1.31x** | n/a |
| **Result OK 传播 4 层(裸)** | 0.748 / 0.185 / 1.073 | 0.945 / 0.185 / 1.074 | 0.551 / 0.184 / 1.074 | Win **WIN 1.26x**；Lin/M4 **TIE** | Win 裸路径慢于 std（见下假象说明）；Lin/M4 **TIE** |
| **Result OK 传播(解包消费,真实)** | 1.450 / 0.331 / 1.588 | n/a（bench 未对 absl 消费变体） | 1.423 / 0.330 / 1.588 | n/a | 三平台实质 **TIE**（Win sign 0.981 可忽略） |
| **Result ERROR 上抛 3 层** | 10.23 / 9.35 / 3.17 | 22.00 / 22.21 / 6.80 | 0.553 / 0.225 / 0.989 | 三平台 **WIN**（2.15x / 2.38x / 2.14x） | 对 std 不可比（errc 无堆消息） |

### 空间维度（三平台一致，除 StatusOr ABI）

| 类型 | Win | Lin | M4 |
|---|---|---|---|
| `hyle::Status` | 8 | 8 | 8 |
| `hyle::Result&lt;int&gt;` / `Result&lt;void&gt;` | 16 / 16 | 16 / 16 | 16 / 16 |
| `std::expected&lt;int,errc&gt;` | 24 | 24 | 24 |
| `absl::Status` | 8 | 8 | 8 |
| `absl::StatusOr&lt;int&gt;` | **24** | **16** | **16** |

HYLE `Result&lt;int&gt;=16B` 在 Windows 比 absl StatusOr 省 1/3；POSIX 上两者同宽。

### 当时判定（仍用于观察平台翻转）

1. **结构优势仍成立（真 SOTA 项）**  
   - **code-only 错误**：三平台对 absl 稳赢（inline 码零堆 vs absl 必堆 rep）——最大设计差。  
   - **带消息构造**：三平台对 absl 约 **1.9–2.5x**。  
   - **ERROR 上抛**：三平台对 absl 约 **2.1–2.4x**。  
   - **OK 构造**：三平台对 absl 统计平；对 std 在 Lin/M4 赢、Win 平。

2. **平台翻转项（不可再写“全线赢”）**  
   - **OK 拷贝**：Win/M4 对 absl **sign 输**（0.95x / 0.72x），Linux **sign 赢** 1.26x。  
   - **heap 拷贝**：Win/M4 赢 absl，**Linux sign 输 0.90x**。  
   这两项是 codegen/原子路径平台差，不是契约不公；当前**不判整体非 SOTA**，但总账必须标 ⚠️ 平台翻转，禁止用单平台数外推。

3. **Result OK 真实用法 = 平 std**  
   解包消费变体三平台实质 TIE。裸“构造即析构”microbench 会放大 `Status` 非平凡析构分支（`is_heap()`）；隔离探针 `expected&lt;int,TrivialErr&gt;` 与 std 同档，说明税来自“可拥有堆消息”的 E 类型，不是传播宏。真实代码取值后该分支与后续工作交织，不构成产品缺陷。

4. **不对 std ERROR 路径宣称输赢**  
   `std::error_code` 无堆消息，ERROR 传播 ns 不可与 `Status`/`StatusOr` 比。

5. **旧 2026-06-30 Win 单遍证书作废**  
   旧表“OK 赢 std 1.5x / 全线 ≥ absl”被三平台+sign 检验推翻其中若干行。发布与总账以本表为准。

### 复跑命令

```bash
# 需 -DHYLE_BUILD_BENCH=ON -DHYLE_BENCH_ABSL=ON，ABSL 指向克隆的 abseil-cpp
cmake --build <build> --target hyle_error_bench
scripts/bench-stat.py <build>/bin/hyle_error_bench -n 15 \
  --pair "absl::OkStatus" "HYLE Status::Ok" \
  --pair 'absl::InvalidArgumentError("")' "HYLE Status(code) inline" \
  --pair "absl::NotFoundError(msg)" "HYLE Status::NotFound(msg)" \
  --pair "absl::Status copy (error)" "HYLE Status copy (error)" \
  --pair "absl::Status copy (OK)" "HYLE Status copy (OK)" \
  --pair "std::expected (unwrap + use)" "HYLE Result (unwrap + use)" \
  --pair "absl::StatusBuilder SetPrepend" "HYLE Status::WithContext" \
  --pair "absl::StatusOr (manual check)" "HYLE Result + ASSIGN_OR_RETURN"
```
