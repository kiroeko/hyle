# io-07a：multishot accept（补完 kMore 半成品）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）|
| **状态** | ✅ multishot accept 三后端。真测 MultishotAccept×3 双平台 + WSL TSan 净 |

> **范围**：补完 io-01 在 `Completion.kMore` 上预留的接口——`submit_accept_multishot`：一次 arm 持续吐新连接、每连接一个 CQE 带 `kMore`（免每连接重 arm）。
> **cloc**：+90。

## 为什么是它
一个 busy listener 每来一个连接就要重新 arm 一个 accept，是可测量的 per-connection 开销。io_uring 的 `IORING_ACCEPT_MULTISHOT` 用一次提交换持续投递。io-01 早把 `kMore` flag 放进 `Completion` 接口正是为了不挡这个——io-07a 是兑现那张欠条。

## 核心决策
- **一提交多完成的契约**：每个 completion 带 accepted fd（>=0）+ **`user_data` 原样返回**（每次同一 token），`kMore` SET 表示仍 armed；`kMore` CLEAR 是终结 completion（multishot 结束——被取消或 fatal error）。调用方**在见到 kMore clear 前不得回收** waker/slot，且必须准备好一次提交来多个完成。
- **三后端统一契约，emulation 诚实**：
  - **io_uring**：原生 `IORING_ACCEPT_MULTISHOT`（填 `sqe.ioprio`，`submit_one` 扩 ioprio 参）。
  - **IOCP**：无原生 → `AcceptEx` 循环 emulation（reap 成功后重 arm + 盖 `kMore`）。
  - **epoll**：一次性 watch + reap 重 arm。
  - IOCP/epoll 是 emulation，但**对上呈现同一 kMore 流契约**，net/ 只见一套统一接口。

## 对抗式自审（正确性 / TSan / 诚实）
- **① 正确性**：MultishotAccept×3（uring/epoll/iocp，一次 submit 服务 4 连接）双平台。
- **④ TSan**：净。
- **诚实标注**：IOCP/epoll 明确是 re-arm emulation，不假装原生。

## 交付物
- `include/hyle/io/proactor.h`（`submit_accept_multishot` 签名 + kMore 流契约注释）
- `src/io/proactor.cc`（io_uring ACCEPT_MULTISHOT + IOCP `iocp_arm_accept`（multishot 标记 + reap 重 arm）+ epoll 重 arm）
- `tests/io/proactor_test.cc`

## 参考
- io-design「② 参考库审计真缺口」io-07a 节；io-01 的 `Completion::kMore` 接口预留。liburing multishot accept。相关里程碑：io-01（kMore 接口）、io-07b（kBuffer + multishot recv）。
