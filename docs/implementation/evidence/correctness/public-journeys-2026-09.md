# 跨模块公开 API 正确性旅程（2026-09）

本页是
[`CLOSED 组件全库正确性资格审计`](../../tasks/project/closed-components-correctness-qualification-2026-09.md)
的 Q08 事实证据，不创建第二套模块状态，也不替代各 factor 的原子语义矩阵。五条旅程只使用公开
C++ API；测试支撑只包括进程 child fixture 与内存中的测试证书。

## 输入身份与统一运行

- candidate：`17125de9a527a3015faa48d69b16da73e1484076`；tree
  `d44f8254d79a0100c6b7cee300e966ef5706c1a6`；baseline `9cf1de33`；
- W1-WIN：MSVC 19.50.35728、Release、shared、WERROR、`HYLE_TESTING=OFF`；
- 产品选择：`container,simd,math,mem,error,obs,sched,io,process`，由中央依赖图闭合；
- feature：OpenSSL 3.6.3、`HYLE_IO_TLS=ON`、nghttp2 1.70.0、`HYLE_IO_HTTP2=ON`；
- 独占 build：`C:\public\project-hyle\_validation\closed-correctness-q08-all-worktree-v1`；每项
  CTest 有 60 秒硬 timeout，网络/文件/进程操作另使用有意义的 deadline/cancel；
- `hyle_correctness_tests` 从零构建 201 个必要产品/测试目标成功；`ctest -L '^integration$'`
  为 **5/5 PASS**，实际测试时间 0.64 秒。该 build 的已测试源码与随后提交的 candidate 字节一致。

```powershell
$env:OPENSSL_ROOT_DIR='C:\public\vcpkg\installed\x64-windows'
scripts/hyle-windows.ps1 -Action Configure -Preset windows-ninja `
  -BinaryDir C:\public\project-hyle\_validation\closed-correctness-q08-all-worktree-v1 `
  -Component 'container,simd,math,mem,error,obs,sched,io,process' `
  -DependencySourceCache C:\public\project-hyle\.deps `
  -DHYLE_BUILD_TESTS=ON -DHYLE_BUILD_BENCH=OFF -DHYLE_WERROR=ON `
  -DHYLE_STATIC=OFF -DHYLE_IO_TLS=ON -DHYLE_IO_HTTP2=ON
scripts/hyle-windows.ps1 -Action Build `
  -BinaryDir C:\public\project-hyle\_validation\closed-correctness-q08-all-worktree-v1 `
  --target hyle_correctness_tests --parallel 2
scripts/hyle-windows.ps1 -Action Test `
  -BinaryDir C:\public\project-hyle\_validation\closed-correctness-q08-all-worktree-v1 `
  --output-on-failure -L '^integration$'
```

## 逐旅程语义矩阵

| Journey | 正常/边界与真实组合 | failure/cancel/ownership/teardown | 裁决 |
|---|---|---|---|
| 数据准备 | strict Base64 exact-size decode → checked bit sizing → owned host Buffer → SIMD int8 dot；结果 204 | malformed input 明确失败；map/flush/close/reset 后 owner 释放 | `PROVEN` |
| 异步运行时 | bounded queue → 两 worker Scheduler/Fiber → timer → context Status → ConcurrentCounter | cross-worker CancelToken、queue drain、Scheduler shutdown 与最终指标一致 | `PROVEN` |
| 模型文件生命周期 | exact Watch → local Transfer atomic publish → async File read → FileMapping 校验 | 缺失源 transfer 失败且旧发布目标不变；watch/file 有 deadline；mapping/file/watcher 显式 reset/close | `PROVEN` |
| 服务连接 | TCP/TLS ALPN `http/1.1` → HTTP Upgrade/WebSocket message；另一 TLS ALPN `h2` 会话双向传输 96 KiB body | 4 KiB stream/65,535-byte connection window 证明 flow-control progress；WebSocket 正常 close；另一个连接的在途 read cancel 导致确定失效和 peer teardown；HTTP/2 server 显式 close | `PROVEN` |
| 工作进程 | ProcessRuntime spawn → scheduler-bound LocalStream stdout 读取 `ready` | pre-cancel 只取消 wait、不误杀 child；force-kill tree、forced wait/reap、pipe EOF/close、active count 归零、runtime/Scheduler shutdown | `PROVEN` |

模型文件旅程没有把 deadline 强加给 Windows 上由同步 Platform handle `adopt` 的 local Transfer：该 handle
不是 `OVERLAPPED`，产品明确拒绝伪造可取消 backend。此格由缺失源 failure、原子发布保持、CTest 硬 timeout
和原生 Watch/File deadline 分别观察，不写成 Transfer cancel `PROVEN`；其原子 cancel/backend 证据仍在
Wave E，当前 Windows backend 刷新归 Q09。

WebSocket framing 开始后的 transport cancel 不是可恢复暂停：实现可能已消费半个 frame，连接依法进入
terminal failed state。Q08 将 clean close 与 cancel teardown 放在两个独立 TLS connection 中，并同步修正
用户指南；没有为让测试通过而弱化协议状态机。

## Inventory 与裁决

`correctness-inventory.py --revision WORKTREE --check` 把跨 factor journey 单独列为 project-owned tests，
不计入任何 factor 的测试总数。结果为 29 个 CLOSED factor、173 个 factor test source，以及 5 个 project
journey source / 1,082 行 / 5 个 declaration，registration/ownership issue 为 0；focused tooling test 通过。

Q08 裁决为 **COMPLETE**：五条公开纵切全部在同一 shipping shape 通过，当前产品 `DEFECT=0`，没有模块
reopen trigger。Windows/Linux/macOS backend、sanitizer、source/static/shared 和 CI 产品投影仍诚实保留给
Q09，不能由本页的 Windows shared run 冒充。
