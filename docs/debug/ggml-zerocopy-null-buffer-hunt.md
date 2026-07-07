# ggml 零拷贝 hunt：从「NULL buffer」到「外部显存注入在 CUDA 上结构性死路」（两阶段，已结案）

> 调查日志（已结案）。这是 compute-05 找回正确设计的**决定性调试**——历时两阶段、5+ 次失败尝试、裸 ggml 最小复现 + WSL gdb 逐行坐实，最终**推翻了 compute-design 发现 2「HYLE 显存零拷贝注入 ggml 算子」的整个愿景**，定下 [`../architecture/borrowing-ggml-memory-model.md`](../architecture/borrowing-ggml-memory-model.md) 的边界：**host 归 HYLE、device 归 ggml，就像 llama.cpp。**
>
> 关联：[`ggml_bridge.cc`](../../src/compute/ggml_bridge.cc)、[`ggml_bridge_test.cc`](../../tests/compute/ggml_bridge_test.cc)、compute-design「发现 2」（被本篇彻底修正）、[`../architecture/borrowing-ggml-memory-model.md`](../architecture/borrowing-ggml-memory-model.md)（本篇结论的正式化）。

## 0. 结论先行

**借 ggml 算子时，无法把 HYLE 自己 cudaMalloc 的显存零拷贝喂给 ggml 的计算张量——CUDA 后端结构性否决**（`buffer_from_host_ptr=false`，`ggml-cuda.cu:5032`）。任何"给 ggml 计算张量注入外部 `->data` / 造假 buffer 元数据"的手法，在**多算子 + 变形状**图上必崩（gdb 坐实崩点 `ggml-cuda.cu:1924`，`src1->buffer->context==NULL`）。正确形态 = **ggml 用 gallocr 分配它算子用的所有 device buffer（含输入），HYLE 的 host 数据经 `ggml_backend_tensor_set` 拷入 / `tensor_get` 拷出**——即 llama.cpp 的做法。host 侧对 ggml 零约束（`tensor_set` 就是裸 `cudaMemcpyHostToDevice`，`ggml-cuda.cu:688`），所以 HYLE mem 全程管 host。

分两阶段查清：

- **阶段 1（`->buffer=NULL` → 段错误）**：发现 2 初版说"`->buffer` 留 NULL 即可零拷贝"——被单算子 GPU 真跑证伪。曾以为修法是 `alloc_ctx_tensors` 拿合法 buffer + 覆写 `->data`（提交为 4135d2f，单发 + 合批 5/5 过）。
- **阶段 2（合批 varied-shape → 更深的崩溃 + 结构性死路）**：为消除 4135d2f 的 throwaway 满量分配去优化（dummy buffer + gallocr），合批在 CUDA 上反复崩。**5+ 次失败后上 gdb，裸 ggml 最小复现坐实**：崩因不是 HYLE 集成 bug，是「size-0 dummy(cudaMalloc(0)=null) + 多输入共享 + varied 形状」触发 ggml-cuda 读被 gallocr 弄丢的 buffer context。进一步查 `buffer_from_host_ptr=false` → **外部显存注入在 CUDA 上根本没有公开 API，此路彻底断**。

## 1. 阶段 1：`->buffer=NULL` 单算子段错误（发现 2 初版半错）

compute-05 首次在 RTX 4070 真跑 `MatmulDefaultDevice`，进程崩于 `0xC0000005`。发现 2 初版断言"ggml 普通 CUDA matmul 只读 `tensor->data`、不查 `tensor->buffer`，故 `->buffer` 留 NULL 即可"——**半错**：

| 行 | 代码 | 读 buffer 干什么 | NULL 时 |
|---|---|---|---|
| 1832-33 | `src1/dst->buffer->context` | 取 `->device`（GPU id）| 空解引用 💥 |
| 1853 | `src0->buffer->buft` | `is_cuda_split` 判断 | 空解引用 💥 |
| 1931/51/73 | `(char*)src0/1/dst->data` | **数据基址**（这个我控制）| — |

数据基址取 `->data`（对），但 `->buffer` 被无条件当元数据源解引用（初版漏看）。CPU 后端不碰这些字段，故 host 推演没暴露。**教训：grep "assert.*buffer" 只证伪"有没有校验"，证伪不了"有没有解引用"；后者要通读函数体每一处 `->buffer`。**

阶段 1 的收场（后被阶段 2 推翻）：`alloc_ctx_tensors(ctx, be)` 让 ggml 给所有 tensor 分配合法 buffer，再覆写 `->data` 指 HYLE 显存。提交 4135d2f，三平台单发+合批 5/5。**但这版有 red-team 揪出的性能病**：每次调用满量 cudaMalloc(A+B+C) 再 cudaFree（同步点），注释"只借 buffer 结构"是假的——它真分配了全部数据显存。为消除这个浪费，进入阶段 2。

## 2. 阶段 2：合批 varied-shape 崩溃——5+ 次失败与失败循环纪律

为省掉 throwaway 分配，改成"输入挂 size-0 dummy buffer（只借元数据）+ 结果走 gallocr"。**单发 CUDA 过；合批（≥2 算子）CUDA 崩，CPU 全过。** 我试了 4 种 buffer 策略（全 dummy / alloc_ctx_tensors / 纯 gallocr / dummy输入+gallocr结果），CUDA 合批全崩。

**误诊排除（诚实记录岔路）**：
- 岔路 A：疑 spawn_blocking 跨线程传指针——排除（崩在 cuda_init 后，时序对不上）。
- 岔路 B：疑测试 harness 的 `std::vector<Buffer>` push_back realloc——排除（reserve 了，device_ptr survive move）。
- 岔路 C：疑 fiber 栈溢出、疑 mimalloc 覆盖 malloc、疑跨线程 CUDA context、疑 galloc 复用/漏 reserve——**逐一用裸 ggml 最小复现排除**。

**关键纪律转折**：Windows 的 `0xC0000005` 是黑盒，stderr probe 在 pool 线程/fiber 硬崩时根本不 flush（`[P] enter` 都印不出）——**"读了源码 + 加 probe"这套在这里失效**。按"失败两次就退一步"的纪律，停止瞎试，改用两个决定性工具：**(1) 裸 ggml 最小复现（脱离 HYLE 层）+ (2) WSL gdb（Linux 符号，真栈）。**

## 3. 决定性实验：裸 ggml 最小复现 + 变量二分

在 WSL/RTX 4070 上写 25 行裸 ggml 复现（无 HYLE 任何层），逐步逼近 bridge 写法：

| 复现 | 配置 | 结果 |
|---|---|---|
| repro3 | dummy 输入 + gallocr 结果 + **count=2 uniform** | ✅ 不崩 |
| repro（跨线程）| backend 主线程建、compute 另一线程跑 | ✅ 不崩 |
| repro（复用 galloc）| 单发图→双发图复用同 galloc | ✅ 不崩 |
| repro5 | **count=3 varied 形状**（{8,16,4},{4,8,6},{12,4,3}）| 💥 崩，同崩点 |
| 二分 | count=3 **uniform** | ✅ 不崩 |
| 二分 | count=2 **varied** | 💥 崩 |

**决定性隔离：崩不崩与 count 无关，与「形状是否一致」有关。** uniform 任意 count 都过，varied 必崩。而且**裸 ggml 就能复现**——不是 HYLE 集成 bug，是 ggml 行为。（`GGML_CUDA_DISABLE_GRAPHS=1` 关 CUDA graph 仍崩，排除 CUDA graph 捕获路径。）

## 4. gdb 坐实到行 + 根因

给 ggml-cuda 加 `-lineinfo` 重编，gdb 抓真实栈：

```
崩点：ggml-cuda.cu:1924   const bool src1_on_device = id == src1_ctx->device;
gdb：  src1->buffer      = 0x555555c3b810   （dummy，非 null）
       src1->buffer->context = 0x0          （NULL！）→ ->device 空解引用
       src1->name        = "leaf_3"          （确认是共享 dummy 的输入叶子）
       repro8: dummy base = (nil)            （cudaMalloc(&p,0) 返回 null）
```

**完整因果链（全 gdb/repro 实证，非猜）**：
1. `cudaMalloc(&ptr, 0)` 返回 **null** → size-0 dummy 的数据基址是 nil。
2. **一个 dummy 被多个输入 tensor 共享**（合批时 4-6 个输入都 `->buffer=dummy`）。
3. `ggml_gallocr_alloc_graph` 处理"被多 tensor 共享的 size-0 dummy"时，**把它的 `->context` 弄成 null**（repro8 证：alloc 前 context 有效、compute 时已 null）。
4. **形状决定走哪条内核路径**：uniform 走 cuBLAS batched 路径**不读** `src1->buffer->context`（所以 uniform 过）；varied 走通用 `ggml_cuda_op_mul_mat`，在 `:1924` 读 `src1_ctx->device` → 读到被 null 的 context → 段错误。

一句话：**根因不是"连续显存交给 ggml"本身，而是"用 size-0 dummy 给输入造假 buffer 元数据 + 多输入共享它"这个手法，gallocr 在多节点时把共享假 buffer 的 context 弄丢，varied 形状的内核路径一读就崩。**（这也解释了全部现象：单发过、CPU 过、uniform 合批过、varied 合批崩、关不关 CUDA graph 都崩。）

## 5. 从「修 dummy」升级到「结构性死路」——为什么不该继续硬凿

修 dummy 的 context 只是治标。退一步问根本：**ggml CUDA 到底支不支持外部显存指针？** 查 device caps：

```cpp
// ggml-cuda.cu:5032   CUDA device 能力表
/* .buffer_from_host_ptr = */ false,
```

对比 CPU 后端有 `ggml_backend_cpu_buffer_from_ptr`（`ggml-backend.h:430`）能零拷贝包外部内存——**CUDA 明确关成 false，无任何公开 API 把外部显存注册成 ggml buffer**。llama.cpp 在 GPU 上也从不注入外部显存：权重加载时拷进 ggml 显存一次、常驻复用（`llama-model.cpp:1570` `alloc_ctx_tensors_from_buft`），激活走 gallocr。**"外部显存注入"在 CUDA 上不是"没找对 API"，是 ggml 结构性不支持。** 硬凿（手工造 buffer 结构、填 context）= 走在 ggml 内部实现的钢丝上，单/多节点/uniform/varied 行为都不一，脆弱且违背"借人家算子就用人家显存规矩"。**此路封死。**

## 6. host 侧是自由的——正确形态

好消息在入口：`tensor_set` 对 host 源指针零约束（`ggml-cuda.cu:688`）：
```cpp
cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, ...);
```
那个 `data` 任意 host 指针都行——**可以是 HYLE mem / PinnedStaging**。所以正确形态（= 阶段 1 那个被冷落的 copy 路径，现在扶正为唯一路径）：

```cpp
ggml_set_input(gA); ggml_set_input(gB);      // gallocr 给输入分配真 buffer（合法 context）
ggml_tensor* gC = ggml_mul_mat(ctx, gA, gB);
ggml_set_output(gC);
ggml_gallocr_alloc_graph(galloc, gf);         // device buffer 全归 ggml
ggml_backend_tensor_set(gA, host_a, ...);     // HYLE host 数据拷入（H2D）
ggml_backend_tensor_set(gB, host_b, ...);
ggml_backend_graph_compute(be, gf);
ggml_backend_tensor_get(gC, host_c, ...);     // 结果拷回 HYLE host（D2H）
```
**device 归 ggml（gallocr），host 归 HYLE（tensor_set/get 的源/目标是 HYLE mem）。** 边界正式化见 [`../architecture/borrowing-ggml-memory-model.md`](../architecture/borrowing-ggml-memory-model.md)。

## 7. 验证

- **RTX 4070（Win MSVC+nvcc）**：4/4 PASSED，**含 BatchedMatmul（varied 形状 {8,16,4}/{4,8,6}/{12,4,3}）**——正是阶段 2 崩的那个 case，现在 gallocr 模型下 buffer context 合法，跑通。真发构建（4/4 obj 全新编）+ 真跑（fresh cuda init + 计时），非 stale exe。
- **WSL GCC14+nvcc / mac host**：结案时同步验证（见 compute-05 spec 状态）。
- 测试演化：旧 5 测试删掉 `CopyPathMatchesZeroCopy`（现在只有一条路，copy-vs-zerocopy 对拍无意义）→ 4 测试；红队审计后补 `BackendReuseAcrossShapes`（持久 gallocr create-once/reuse-many，覆盖旧崩溃的跨调用变形状路径）→ 5 测试。合理演化，非偷工。

## 8. 教训

1. **"读了源码"≠"读全了"**：grep 特定模式只覆盖你想到的问法；解引用崩溃要通读每一处 `->field`。发现 2 两次都栽在只 grep 了一半。
2. **CPU 能过不代表 CUDA 能过**：后端对同一 tensor 字段的依赖不同。跨后端契约差异必须每后端真跑，host 推演不能替代。
3. **黑盒崩溃换工具，别硬凿 probe**：Windows `0xC0000005` + pool 线程 stderr 不 flush = probe 失效。裸复现（脱离集成层定位是不是自己的 bug）+ Linux gdb（真栈 + `-lineinfo` 逐行）才是决定性手段。5 次失败尝试的教训：**失败两次就该换调试手段，而不是换实现方案继续试。**
4. **变量二分**：uniform-vs-varied 这个"最后未复制的变量"是钥匙。复现时逐一对齐真实触发条件（我早期复现全用 uniform，所以一直不崩，误判成 HYLE bug）。
5. **文档先行的"定案"也是待证伪的假设**：发现 2 写得笃定（带行号）仍两次半错。真跑是最终裁判。

## 9. 关联修正（本轮闭环回填）

- compute-design「发现 2」+「compute-05 定案」：删除"零拷贝显存注入"愿景，改为 host/device 归属边界，指向 `borrowing-ggml-memory-model.md`。
- compute-05 spec ①：从"零拷贝 device_ptr 注入"改为"host 归 HYLE / device 归 ggml gallocr"。
- 新增 `../architecture/borrowing-ggml-memory-model.md`：边界的正式信源。
- 代码：`ggml_bridge.cc` 收敛为 gallocr + tensor_set/get 单一路径；API 收 host `float*`（可 pinned），删 device-Buffer 注入 + dummy buffer。
