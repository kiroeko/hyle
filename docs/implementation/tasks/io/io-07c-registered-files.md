# io-07c：registered/fixed files（IOSQE_FIXED_FILE）+ direct-accept 原语

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）|
| **状态** | ✅ proactor 原语已做 + 实测边际/无收益 → 主动不改 TcpStream（io_uring only，IOCP/epoll 透明回退裸 fd）。proactor_test 验全路径 |

> **范围**：`register_file_table` + `assign_fixed_slot`/`release_fixed_slot` + `submit_recv_fixed`/`submit_send_fixed`（fixed-file）+ `submit_accept_direct`/`submit_close_direct`（direct-accept）。**关键交付是一个诚实的性能结论：原语做齐、实测边际，主动停在原语层不改核心 RAII。**
>
> **2026-07-20 API 收口**：全部原语已移入显式 `IoUringExtension`；
> portable `RecvOp/SendOp` 不再含 `fixed_slot` 暗门。release 现在返回 bool，
> 只有内核 `FILES_UPDATE` 成功才回收本地槽。direct accept 的 CQE 槽也已进入
> 本地 mirror（`-2`），close 成功前不会被 `assign_fixed_file` 覆盖；手写
> fixed/direct/multishot SQE 与普通提交共用满环 flush 防线。
> **cloc**：proactor +~340（register_file_table/assign/release/recv_fixed/send_fixed/accept_direct/close_direct）（+ proactor_test）。

## 为什么是它
参考库审计发现 io_uring 的 registered-file 能力没抄。理论上：把 socket 放进注册文件表后用**槽索引**引用（`IOSQE_FIXED_FILE`），内核免每 op `fget/fput`——高 op 率 / 多并发 socket 时的小额 per-op 节省。direct-accept 更进一步：accept 直落 registered 槽，连接**从不是进程 fd**。值得做原语；但值不值得改 TcpStream 拥有语义，要实测说话。

## 核心决策与实测
### registered/fixed files（第一步）
`register_file_table`（sparse `IORING_REGISTER_FILES2`）+ `assign_fixed_slot`/`release_fixed_slot`（线性找空槽 + `FILES_UPDATE` 填/清）+ `submit_recv_fixed`/`submit_send_fixed`（`IOSQE_FIXED_FILE` + 槽索引）。raw fd 仍归调用方所有，这只映射表项。
- **实测**：socketpair ping-pong fixed vs raw fd **+0.8-3.3%**（3 跑一致为正，但淹在 err% 边缘）。

### direct-accept（第二步）
`submit_accept_direct`（`file_index=IORING_FILE_INDEX_ALLOC`，连接直落 registered 槽、CQE res=槽索引、从不是进程 fd）+ `submit_close_direct`（`IORING_OP_CLOSE` with file_index）。2026-07-20 后 fixed 收发只存在于显式 extension，不再渗入 portable `RecvOp/SendOp`。
- **实测（拆相追根 0.7% 问题）**：合并 bench direct 稳定慢 ~1%，但**拆相后 accept 原语本身 direct 反而快 ~10-30ns/conn**（省内核 fget/fput）。那 ~1% 全是 microbench 即建即关的 **teardown 不对称伪影**（direct 的 close_direct + reap + FILES_UPDATE 2 syscall 其一冗余 vs normal `::close` 1 syscall），生产不 accept-完-立刻-close 故不出现。

### 结论：停在原语层
accept 机制的赢面 ~10-30ns 被 TCP 握手 ~330µs 淹没。**为已证微弱的收益改核心 RAII = 纯风险**，主动不做 TcpStream 拥有语义改造。原语留供未来高建连速率 / 多核争 fd 表场景。IOCP/epoll 不暴露该 extension；调用方在 typed Unimplemented 后选择 portable 裸 fd 产品腿。

## 对抗式自审（正确性 / 性能 / 过度复杂）
- **① 正确性**：`RegisteredFixedFiles` 与小环深度批量提交在当前 WSL
  io_uring 真跑；`DirectAccept` 在支持该变体的内核验全路径，当前 WSL 内核
  对 CQE 返回 `-EINVAL` 时明确 skip，不伪报通过。Windows/M4 验证扩展不可取得。
- **② 性能**：诚实拆相——不把 teardown 伪影误报成 accept 回退，也不把 +0.8-3.3% 吹成收益。
- **⑥ 过度复杂**：主动停在原语层，不投机改 RAII——「已证微弱收益 vs 纯风险」是这条里程碑的核心判断。

## 交付物
- `include/hyle/io/proactor.h` + `src/io/proactor.cc`（register_file_table / assign_fixed_slot / release_fixed_slot / submit_recv_fixed / submit_send_fixed / submit_accept_direct / submit_close_direct）
- `include/hyle/io/net.h`（RecvOp/SendOp 的 `fixed_slot` 字段 + submit_recv_fixed/send_fixed 分支）
- `tests/io/proactor_test.cc`

## 参考
- io-design「② 参考库审计真缺口」io-07c 节 + direct-accept 拆相节。liburing registered files / direct descriptors。相关里程碑：io-04（TcpStream/RecvOp）、io-07b（同期 io_uring 高级能力）。详见 [performance/benchmarks-io.md](../../evidence/performance/benchmarks-io.md) direct-accept 拆相节。
