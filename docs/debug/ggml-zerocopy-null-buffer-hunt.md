# ggml 零拷贝 NULL-buffer 崩溃 Hunt（finding 2 被真跑证伪 —— GPU 段错误、已修）

> 调查日志（已结案）。触发：compute-05 ggml_bridge 首次在 RTX 4070 真跑 `GgmlBridge.MatmulDefaultDevice`，进程崩于 `0xC0000005`（访问违规）。
> 关联：[`ggml_bridge.cc`](../../src/compute/ggml_bridge.cc)（`matmul_blocking`，修复点）、[`ggml_bridge_test.cc`](../../tests/compute/ggml_bridge_test.cc)、compute-design「发现 2」（被本篇修正）。
> 结论先行：**compute-design 发现 2 断言"ggml 普通 CUDA matmul 只读 `tensor->data`、不查 `tensor->buffer`，故 no_alloc + 手设 data、`->buffer` 留 NULL 即可零拷贝"——这个断言半错。** ggml CUDA matmul 路径**无条件解引用 `tensor->buffer`**：`src0->buffer->buft`（split 判断，`ggml-cuda.cu:1853`）、`src1/dst->buffer->context->device`（取 GPU id，`:1832-1833`）。`->buffer=NULL` → 空解引用 → 段错误。**但发现 2 的另一半是对的**：kernel 的数据基址确实取 `src0/1/dst->data`（`:1931/1951/1973`），那个我控制。故修法不是"放弃零拷贝改拷贝"，而是"让 ggml 分配 tensor 拿到合法 `->buffer` 元数据，再覆写 `->data` 指向 HYLE 显存"——真实 matmul 数据仍零拷贝。

## 0. 为什么这篇值得独立成文：文档调研的"看了源码却看漏一半"

发现 2 是**读了 ggml-cuda.cu 源码后写下的**——不是凭空猜的。它 grep 到"普通算子不校验 buffer 归属（ownership）"，这句本身没错（`is_cuda_split` 那些 assert 确实只在 split-buffer 路径）。**错在把"不校验归属"外推成"根本不解引用 buffer"**——漏看了同一函数里为拿 device-id 和 split-flag 而做的**无条件 `->buffer` 解引用**。教训：grep "assert.*buffer" 只能证伪"有没有校验"，证伪不了"有没有解引用"；后者要通读函数体的每一处 `->buffer`。**文档先行的调研也会看漏，真跑是最终裁判。**

## 1. 现象：GPU 找到了，matmul 一算就崩

```
[ RUN      ] GgmlBridge.MatmulDefaultDevice
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 12281 MiB):
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
TEST-EXIT=-1073741819        # = 0xC0000005 访问违规
```

关键观察：`ggml_cuda_init` 打印**之后**才崩——即后端初始化成功（backend/stream/context 都好），崩在**真正提交 matmul 计算**那一步。排除了环境/链接/初始化类问题，锁定"计算路径解引用了某个空指针"。

## 2. 误诊排除（诚实记录走过的岔路）

- **岔路 A：先怀疑 spawn_blocking 跨线程传指针出错。** 排除：崩点在 `ggml_cuda_init` 之后，而 pool 线程闭包里第一件事就是 `ggml_init`（元数据 arena）——若指针传递错，`ggml_init`/`ggml_new_tensor_2d` 阶段就该崩，不会等到 `ggml_cuda_init`（那是 backend create 时打印的、更早）。时序对不上。
- **岔路 B：怀疑 golden 参考索引错导致越界。** 排除：那会是"结果错"（对拍失败返回 -8），不会是段错误；且崩在 compute 内、根本没到对拍。
- **回到源头：通读 `ggml_cuda_mul_mat` 函数体的每一处 `->buffer`**（而非只 grep assert）。

## 3. 决定性证据：源码里三处无条件 `->buffer` 解引用

`ggml-cuda.cu` 的 matmul 路径（`ggml_cuda_mul_mat`，约 `:1810-1990`）：

| 行 | 代码 | 读 buffer 干什么 | NULL 时 |
|---|---|---|---|
| 1832 | `src1->buffer->context` | 取 `->device`（GPU id，`:1924`）| 空解引用 💥 |
| 1833 | `dst->buffer->context` | 取 `->device`（`:1925`）| 空解引用 💥 |
| 1853 | `src0->buffer->buft` | `is_cuda_split` 判断 | 空解引用 💥 |

**而数据基址**（`:1931/1951/1973`）：
```cpp
dev[id].src0_dd = split ? src0_extra->data_device[id] : (char*)src0->data;  // 非 split → ->data
dev[id].src1_ddf = (float*)src1->data;
dev[id].dst_dd   = (float*)dst->data;
```
→ **非 split 路径，kernel 直读 `->data`**。所以 `->buffer` 只被当"元数据源"（device id + split flag），不当数据源。发现 2 的"数据看 ->data"对，"不碰 ->buffer"错。

## 4. 根因

`matmul_blocking` 建 no_alloc tensor 后只设了 `->data`，`->buffer` 保持 `ggml_new_tensor_2d` 给的默认 `NULL`。CUDA matmul 第一步 `src1->buffer->context`（`:1832`）即对 NULL 解引用。CPU 后端恰好不崩（它的 matmul 不走这些 `->buffer` 解引用），所以这是 **CUDA 专属**触发——也解释了为何 host-only 的思路推演阶段没暴露。

## 5. 修法：ggml 分配拿合法 buffer 元数据 + 覆写 data 保零拷贝

```cpp
ggml_tensor* gA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
ggml_tensor* gB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
ggml_tensor* gC = ggml_mul_mat(ctx, gA, gB);

// 关键新增：让 ggml 给 ctx 里所有 tensor 分配（gA/gB/gC 拿到合法 ->buffer）。
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

// 覆写数据基址为 HYLE 显存（ggml 那份分配的字节被弃用，只借它的 buffer 元数据结构）。
gA->data = a_dev;  gB->data = b_dev;  gC->data = c_dev;

ggml_backend_graph_compute(be, gf);
ggml_backend_buffer_free(buf);   // 释放 ggml 那份 throwaway 分配（非 HYLE 内存）
```

- `->buffer->buft`/`->context->device` 现在合法（ggml 分配填的），split 判断/device-id 都能读；
- kernel 数据仍从 `->data`（HYLE 显存）取——**真实 matmul 数据零拷贝不变**；
- 代价：ggml 多分配一份 m·k+k·n+m·n 的显存立即弃用（元数据副产品）。可接受；未来若嫌浪费，可探索手工造 buffer 结构（内部 API，脆弱，暂不做）。

## 6. 验证

- **RTX 4070 真跑**：`GgmlBridge.MatmulDefaultDevice` OK（6.2s 含 cuda init），zero-copy CUDA matmul + spawn_blocking + fiber 挂起唤醒 + golden 对拍逐元素通过。
- `CpuThreadModelSplit`（n_threads=1 与全核）OK；`ConcurrentIndependentBackends`（6 fiber 各自 backend 并发）OK。
- 三平台 + TSan 待跑（本篇结案时 Windows 已过）。

## 7. 教训

1. **"读了源码"≠"读全了"**：grep 特定模式（assert/ownership）只覆盖你想到的问法；解引用崩溃要通读函数体每一处 `->field`。发现 2 就栽在只 grep 了 assert。
2. **CPU 能过不代表 CUDA 能过**：后端间对同一 tensor 字段的依赖不同（CPU matmul 不碰 `->buffer`，CUDA 碰）。跨后端契约差异必须真跑每个后端，host 推演不能替代。
3. **文档先行的"定案"也是待证伪的假设**：发现 2 写得很笃定（还带行号），仍半错。真跑是最终裁判——这也是 CHECKLIST "真编真测（stale exe 会假 PASS）"的价值：第一次崩溃恰恰证明 exe 是新的、测试是真的。

## 8. 关联修正

- compute-design「发现 2」需回填修正：NULL-buffer 零拷贝证伪，改为"alloc_ctx_tensors 拿合法 buffer 元数据 + 覆写 data"。（本轮闭环回填。）
