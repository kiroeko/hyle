# IO Transfer v1：可靠模型/制品传输工作流

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_transfer` / IO maintainers / `io-transfer-v1`；task kind：`NEW`；
  baseline commit：`e1499bb404e1a69881fb1c4d91c60aa183d1d1a7`；
- demand：`IOTRANSFER-MODEL-01`、`IOTRANSFER-RESUME-02`、`IOTRANSFER-ATOMIC-03`。
  llama.cpp 与推理部署普遍需要下载/复制大型模型；当前 HYLE 只有 file/socket primitive，用户必须
  自写 redirect/range/resume/checksum/temp-file/atomic publish 和失败清理；
- normal journey：在总 deadline/cancel 和 byte budget 下，把 HTTP(S) 或本地 file source 可靠地
  传到临时文件，支持安全 resume、进度/速率 backpressure、校验，成功后原子发布；失败不损坏旧文件；
- scope：typed source/destination、bounded copy pipeline、HTTP range/redirect policy、resume validator、
  checksum hook、temp-file + fsync + atomic replace、progress observation、cancel/cleanup；
- non-goals：Hugging Face/model repository schema、authentication UI/token storage、archive extraction、
  model parsing/loading、P2P/RDMA、object-store SDK、route/server、Native FFI/C API；
- dependencies / consumers：候选 `PUBLIC io_core,io_http,error,mem,container_text,time,sched,platform_fs`；
  HTTPS route only when `io_tls` is available。最终 edge 在 `AUDITED` 冻结；
- tier：`P1`。primary 为 256 MiB sparse/generated source→temp→atomic publish，holdout 为 interrupted
  range resume；吞吐预算以 OS/file/network 限制为主，但用户态只允许有界 buffers、无全文件 copy、
  O(1) memory in file size；
- budget：总 8 engineering hours，审计/API 3 小时，实现/correctness 3 小时，P1 2 小时；一次 candidate
  bundle，最多一次机制诊断；
- machines / instruments：`W1-WIN/W1-WSL/M1-MAC`，`closure-correctness-v1` +
  `closure-perf-v1`；
- strong peers：libcurl transfer/resume semantics 与 Boost.Beast composed HTTP operations；它们用于
  workflow/contract 对照，不自动成为相同 scheduler/allocator 的 same-contract peer；
- provenance：llama.cpp model download/remote media 是真实 reference；vLLM/SGLang/ds4 的 model
  deployment 说明制品获取是通用 journey，但仓库/auth policy留在上层；
- API freeze 前明确 partial file identity、resume validator、redirect/security、checksum ownership、
  progress callback lifetime/reentrancy、cancel/close、atomic replacement 和 direct/buffered file policy；
- gates：empty/large/short IO、disconnect/resume/range mismatch/redirect loop/checksum/OOM/disk full/
  cancel/old-file preservation、三平台、source/static/shared、P1、guide、红队；
- stop：若 HTTP factor 尚未提供完整 streaming body/cancel contract，本卡可保持 `AUDITED/BLOCKED`，
  不复制 socket/parser；不把模型 schema、auth 或 batch policy下沉。

## SCOPED audit queue

- 追踪 llama.cpp download 与四组引擎部署路径，区分库机制与应用 repository policy；
- 对 libcurl/Beast 的 redirect/range/resume/temp-file/checksum 生命周期建立能力图；
- 冻结与 `platform_fs`/Mem/HTTP/TLS 的唯一 ownership 和有限 FIX 集；
- 没有 existing public surface；新 API 必须先通过 normal C++ sketch，再允许实现。

## Current decision

`SCOPED / ACTIVE`。本卡只授权审计；尚未进入 `AUDITED`。
