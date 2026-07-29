# error 实测铁证

> 本页是 [performance 总账](./README.md) 的 error 模块详节。稳定设计见
> [error-design](../architecture/error-design.md)，用户选型见
> [error guide](../guide/error.md)。benchmark 只验证消费者会使用的产品形状，不定义 API。

## 当前证书（2026-07-29，三平台 n=15）

基础产品能力基于 `befaad0`，非法 Result 诊断冷热分离基于 `7d22eb8`，渐进 context
表示固定为 `8dace9d`；benchmark 消费形状仍是 `6d42add`，没有为该实现新增测量专用 API。
Abseil 固定到本地参考库 commit `35b9f25b`。环境为 Windows MSVC 19.50、WSL GCC 14.2、
Apple M4 AppleClang 21，Release、`HYLE_TEST_INSTRUMENTATION=OFF`（发布形状），使用
`scripts/bench-stat.py -n 15` 的同进程配对判定。表中绝对值统一按
**Windows / Linux / M4** 排列，不能跨机器横比；倍数是配对样本的中位比值，可能与两列
独立中位数之比略有差异。

本证书严格分成两种消费者信息合同：

1. **不携带额外诊断**：只有成功值或 canonical code。主对手是 Abseil
   `Status/StatusOr` 的同能力路径；`std::error_code/expected` 是不能携带任意消息和 native
   detail 的较弱下界，只单独登记，不用于宣称同能力 SOTA。
2. **携带诊断**：相同 canonical code、owning message、语义 context 或同一
   machine-readable native value。对手是 Abseil `Status/StatusOr/StatusBuilder/payload`。

没有第三类“为了 benchmark 好测”的公开类型。benchmark 也不允许给产品对象增加常驻字段、
缓存或特殊入口。

## 合同一：不携带额外诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| OK 构造并检查 | 0.547 / 0.057 / 0.226 | 0.547 / 0.054 / 0.226 | 三平台 TIE |
| OK Status 拷贝并检查 | 0.526 / 0.280 / 0.236 | 0.551 / 0.276 / 0.257 | Win/Lin TIE；M4 **WIN 1.093x** |
| code-only Status 构造 | 0.547 / 0.060 / 0.224 | 1.694 / 0.916 / 0.958 | **WIN 3.10 / 15.58 / 4.28x** |
| `Result<int>` 经过四个真实调用帧，caller 解包并消费 | 2.585 / 2.478 / 3.058 | 3.181 / 2.915 / 3.415 | **WIN 1.216 / 1.191 / 1.113x** |
| code-only Result 错误经过四个真实调用帧 | 2.380 / 1.103 / 1.443 | 4.240 / 2.258 / 2.455 | **WIN 1.773 / 2.006 / 1.681x** |

四层传播不是在同一个函数里连续创建三个临时 Result，而是 leaf → layer1 → layer2 → caller
的四个真实 `noinline` 调用帧，三家边界完全对称，最后消费成功值。旧形状曾在 M4 同时保活
多个临时对象，制造约 8% 假 LOSE；旧 GCC 形状还只内联了 std/Abseil 的外层包装，制造约
18% 假 LOSE。修复的是测量合同，不是为跑分修改产品 API。

### 能力更弱的 std 下界

| 操作 | HYLE ns（Win/Lin/M4） | `std::error_code/expected` ns | 结论 |
|---|---:|---:|---|
| code-only error 构造 | 0.547 / 0.060 / 0.224 | 0.374 / 0.062 / 0.224 | Windows std **1.46x** 快；Linux/M4 TIE |
| 四帧 Result OK + consume | 2.585 / 2.478 / 3.058 | 2.248 / 2.624 / 3.328 | Windows std **1.15x** 快；Linux/M4 HYLE 约 **1.07/1.09x** 快 |
| 四帧 code-only Result error | 2.380 / 1.103 / 1.443 | 4.836 / 4.413 / 2.166 | HYLE **WIN 2.03 / 4.00 / 1.50x** |

Windows 的成功下界是真实、保留的 LOSE：`std::error_code` 是平凡析构的两个 word，能力比
owning `Status` 小。它适合只需 category/value 的局部代码，但不能代替 HYLE 跨模块需要的
canonical message/native/context carrier。故它是成本下界和选型提示，不是同能力对手。

## 合同二：携带消费者需要的诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| owning message Status 构造 | 27.744 / 9.876 / 17.412 | 47.071 / 20.017 / 31.054 | **WIN 1.704 / 2.037 / 1.877x** |
| owning message Status 拷贝 | 10.238 / 8.850 / 3.181 | 10.580 / 9.262 / 4.180 | **WIN 1.037 / 1.045 / 1.314x** |
| 前置一层语义 context | 44.660 / 21.686 / 17.401 | 259.779 / 91.417 / 120.739 | **WIN 5.800 / 4.203 / 6.440x** |
| 前置三层语义 context | 78.994 / 43.255 / 42.714 | 767.287 / 277.394 / 339.224 | **WIN 9.721 / 6.480 / 7.721x** |
| message Result 错误跨两个调用边界 | 9.709 / 9.221 / 3.814 | 21.157 / 21.248 / 7.202 | **WIN 2.198 / 2.292 / 1.860x** |
| owning 人读格式化 | 31.484 / 19.741 / 23.787 | 40.831 / 22.874 / 30.729 | **WIN 1.289 / 1.165 / 1.232x** |
| 首错聚合 `Update` | 9.221 / 8.794 / 3.181 | 10.537 / 9.188 / 4.184 | **WIN 1.141 / 1.045 / 1.315x** |
| errno → canonical + operation + native value | 39.732 / 51.330 / 26.238 | 168.032 / 74.727 / 116.480 | **WIN 4.224 / 1.451 / 4.608x** |
| message + Vulkan native value 构造 | 35.018 / 12.122 / 16.477 | 87.575 / 37.709 / 53.482 | **WIN 2.500 / 3.110 / 3.373x** |
| Vulkan native value 查询 | 2.017 / 0.642 / 0.962 | 5.683 / 4.636 / 4.422 | **WIN 2.807 / 7.151 / 4.616x** |

Abseil payload 是可携带多项任意数据的通用系统；HYLE native detail 只表达一个固定
`{domain, int64 code}`。上表的 native 行只说明**同样存取一个原生数值**时 HYLE 更快，
不声称 HYLE 具备或击败任意 payload 系统。这个窄形状来自 Vulkan/CUDA/OS 恢复逻辑的真实
需求：此前只有 canonical code + message，消费者不得不解析 message。

同能力 Abseil 项在三平台没有确认 LOSE。这个结论不外推到未上架的 payload、序列化、
backtrace 或异常系统；参考库有而 HYLE 用户没有需求的能力，不会只为扩大 benchmark 表而
进入 error。

## 空间与复杂度

- 64 位平台：`Status=8B`，`Result<int>/Result<void>=16B`；`absl::Status=8B`，
  `absl::StatusOr<int>` 在 Windows 为 24B、Linux/M4 为 16B。
- OK 与 code-only Status 是 O(1)、零分配；native query 是 O(1)。
- message 构造是 O(message bytes)，heap Status copy 是 O(1) 原子 retain。
- 第一层 `WithContext` 精确生成扁平消息；第二层 rvalue 才进入 context 专属几何 prefix
  storage，唯一所有权且容量足够时只写新前缀。连续 rvalue 链的总复制为 O(最终消息字节)，
  扩容 O(log final bytes)；lvalue/共享值保持 COW，单次仍是 O(context + old message)。普通
  message/native Status 不携带 capacity/offset。
- errno canonical 分类是 O(1)；native description 的取得由 OS/CRT 主导，最终只构造一个
  owning Status。
- 非法 Result error arm 的 fail-fast 诊断放在冷 `noinline` helper；正常传播帧不复制 abort
  文本和控制流。M4 栈帧从 112B 降到与 Abseil 相同的 80B，这是冷热路径架构修正，不增加
  产品状态。

## 复跑命令

```bash
# 需 HYLE_BUILD_BENCH=ON、HYLE_BENCH_ABSL=ON，并固定 HYLE_ABSL_DIR。
cmake --build <build> --target hyle_error_bench hyle_error_normal_bench

scripts/bench-stat.py <build>/bin/hyle_error_bench -n 15 \
  --pair "absl::OkStatus" "HYLE Status::Ok" \
  --pair "absl::Status copy (OK)" "HYLE Status copy (OK)" \
  --pair "absl::Status(code,empty,no-loc)" "HYLE Status(code) inline" \
  --pair "absl::StatusOr (unwrap + use)" "HYLE Result (unwrap + use)" \
  --pair "absl::StatusOr code-only error" "HYLE Result code-only error"

scripts/bench-stat.py <build>/bin/hyle_error_normal_bench -n 15 \
  --pair "absl::Status(code,msg,no-loc)" "HYLE Status::NotFound(msg)" \
  --pair "absl::Status copy (message)" "HYLE Status copy (message)" \
  --pair "absl::StatusBuilder SetPrepend" "HYLE Status::WithContext" \
  --pair "absl::StatusBuilder SetPrepend x3" "HYLE Status::WithContext x3" \
  --pair "absl::StatusOr message error" "HYLE Result message error" \
  --pair "absl::Status::ToString owning" "HYLE to_string owning" \
  --pair "absl::Status::Update" "HYLE Status::Update" \
  --pair "absl::ErrnoToStatus + native payload" "HYLE status_from_errno" \
  --pair "absl::Status + SetPayload" "HYLE Status::FromNative" \
  --pair "absl::Status GetPayload + decode" "HYLE Status::native_error"
```
