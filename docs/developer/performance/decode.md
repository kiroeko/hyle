# CUDA 推理性能

更新：2026-11-03，M1-7a 最新上游对照。**目前不能说默认配置全面领先 llama.cpp。**

- 标准 pp512：Hyle 是上游的 **80.9%～88.6%**；标准 tg128 是 **98.7%～105.4%**，四档范围均重叠，decode 胜负未证明。
- 真实八轮：合并普通 decode **+4.5%**（95.343 / 91.234 tok/s），MTP **+14.0%**（120.131 / 105.368 tok/s）。MTP 第4轮 **−23.1%**，第6轮范围重叠。
- 八轮请求级 prefill 合计：普通 **196.040 / 366.660ms**，MTP **198.396 / 448.184ms**。这是现有Hyle入口与上游HTTP服务端的返回计时，不能当成标准pp计算吞吐的结论。
- 外部启动到ready：普通 **2.165 / 1.963s**，Hyle慢约202ms；MTP **2.198 / 2.250s**，范围重叠。
- 专用显存采样峰：普通 **4595 / 4263MiB**（+332MiB），MTP **5069 / 4575MiB**（+494MiB）。私有提交峰另列，工作集不作内存优势比较。

以上均为Hyle / 最新上游。未修改引擎、kernel或数学路径。本页只保留本次最新数据，旧吞吐与归因保存在Git历史 `d7c82a76f`。

同窗冻结版对照：普通合并decode为冻结版的102.7%，范围重叠；MTP为109.9%，范围未重叠。以前普通“小幅领先”的结论在本窗未证明，MTP整体领先仍有支持。

## 版本与口径

RTX4070 12GB、i9-13900K、驱动617.14，Qwen3.5-2B本地BF16权重，greedy。Hyle为默认Release构建 `d7c82a76f`，推理实现沿用 `5192b4232`。没有下载模型权重。

上游固定为 [b12063](https://github.com/ggml-org/llama.cpp/releases/tag/b12063)，提交 `2cbca389f515df455412dafa07e2b1359cc72a6d`，2026-11-03发布的预发布构建。最新稳定tag `v0.8.0`没有Windows CUDA包。本次官方包与自编版使用同一b12063源码。

官方Windows CUDA13.4包：CPU主程序Clang20.1.8；CUDA构建使用发布工作流的`GGML_CUDA=ON / GGML_BACKEND_DL=ON / GGML_NATIVE=OFF / GGML_CPU=OFF / GGML_CUDA_CCCL_VERSION=v3.4.3`。本机自编：MSVC19.50.35728、CUDA13.3.73、Release，`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF`，其余默认。四个配置各跑一次pp/tg、每点`-r5`；八个点TPS的几何平均择快，选中官方包FA=on。官方on/off得分1003.063/807.213，自编on/off为924.915/814.150；这些仅用于选配置，不算三进程验收。

冻结第三方仍是`d9e03f1074-ggml456172e-matched-consumer`，与本次上游、Hyle同窗交错。正式计时36个独立进程（A18、B/C18）；显存另18个独立进程。每格3个新进程，启动前检查竞争测试进程和连续3次GPU利用率≤10%；无3.B、其它推理或profiler并行。Windows WDDM桌面后台仍存在。`llama-bench`每进程重复5次，先取该进程的stock平均TPS，再对3个进程取中位数与范围，内部重复不冒充独立进程。

## 全量对照

表内是 **Hyle / 上游 = Hyle÷上游**。TPS越高越好；ms、s和MiB越低越好。†表示两组独立进程范围重叠；未标记也不代表显著性检验。八轮合并decode用504个后续输出除以8轮decode总时间，不平均每轮TPS。冻结版列是本窗口的中位数。

| 场景 | 指标 | Hyle / 上游 = 比例 | Hyle范围 | 上游范围 | 冻结版中位数 | 独立进程数 |
|---|---|---:|---:|---:|---:|---:|
| pp512 d=0 | prefill_tps | 8941.795 / 11058.278 = 80.9% | 8142.973–10005.100 | 10766.311–11089.927 | — | 3/3 |
| tg128 d=512 | decode_tps | 101.423 / 98.957 = 102.5%† | 94.955–103.867 | 92.658–100.620 | — | 3/3 |
| pp512 d=4096 | prefill_tps | 8946.982 / 10860.520 = 82.4% | 8605.461–9457.907 | 10464.977–11734.514 | — | 3/3 |
| tg128 d=4608 | decode_tps | 96.238 / 94.826 = 101.5%† | 92.953–102.416 | 92.481–98.369 | — | 3/3 |
| pp512 d=16384 | prefill_tps | 7998.863 / 9186.153 = 87.1% | 7627.947–8440.738 | 8823.059–9948.803 | — | 3/3 |
| tg128 d=16896 | decode_tps | 92.773 / 93.958 = 98.7%† | 90.509–98.807 | 91.840–94.112 | — | 3/3 |
| pp512 d=30720 | prefill_tps | 6987.672 / 7889.250 = 88.6% | 6847.648–7219.534 | 7877.519–8481.357 | — | 3/3 |
| tg128 d=31232 | decode_tps | 89.000 / 84.469 = 105.4%† | 87.884–95.056 | 81.055–91.211 | — | 3/3 |
| 8轮 MTP=off 合并 | prefill_ms | 196.040 / 366.660 = 53.5% | 188.563–207.681 | 357.470–371.342 | 184.109 (n=3) | 3/3 |
| 8轮 MTP=off 合并 | decode_tps | 95.343 / 91.234 = 104.5% | 92.746–97.785 | 91.206–91.778 | 92.839 (n=3) | 3/3 |
| 8轮 MTP=off 第1轮/105行 | prefill_ms | 22.778 / 95.039 = 24.0% | 22.645–25.698 | 94.054–96.671 | 32.577 (n=3) | 3/3 |
| 8轮 MTP=off 第1轮/105行 | decode_tps | 97.114 / 92.191 = 105.3% | 95.468–97.334 | 90.698–92.748 | 93.844 (n=3) | 3/3 |
| 8轮 MTP=off 第2轮/108行 | prefill_ms | 23.412 / 37.299 = 62.8% | 22.289–23.916 | 37.235–39.498 | 20.670 (n=3) | 3/3 |
| 8轮 MTP=off 第2轮/108行 | decode_tps | 96.694 / 91.785 = 105.3% | 95.227–97.842 | 91.273–91.791 | 93.586 (n=3) | 3/3 |
| 8轮 MTP=off 第3轮/80行 | prefill_ms | 21.587 / 36.142 = 59.7% | 21.082–24.652 | 35.323–38.436 | 21.662 (n=3) | 3/3 |
| 8轮 MTP=off 第3轮/80行 | decode_tps | 96.573 / 91.818 = 105.2% | 95.684–96.833 | 91.603–92.751 | 91.799 (n=3) | 3/3 |
| 8轮 MTP=off 第4轮/114行 | prefill_ms | 22.698 / 38.315 = 59.2% | 21.036–25.244 | 37.064–42.204 | 21.286 (n=3) | 3/3 |
| 8轮 MTP=off 第4轮/114行 | decode_tps | 95.881 / 91.936 = 104.3%† | 92.056–101.038 | 90.350–92.119 | 94.197 (n=3) | 3/3 |
| 8轮 MTP=off 第5轮/189行 | prefill_ms | 31.015 / 43.138 = 71.9% | 30.643–38.719 | 42.203–44.276 | 26.134 (n=3) | 3/3 |
| 8轮 MTP=off 第5轮/189行 | decode_tps | 96.039 / 90.913 = 105.6% | 95.735–100.220 | 89.443–92.002 | 94.700 (n=3) | 3/3 |
| 8轮 MTP=off 第6轮/110行 | prefill_ms | 24.084 / 37.182 = 64.8% | 21.940–24.331 | 36.919–38.518 | 22.031 (n=3) | 3/3 |
| 8轮 MTP=off 第6轮/110行 | decode_tps | 95.611 / 91.125 = 104.9% | 92.229–97.216 | 90.534–91.763 | 94.443 (n=3) | 3/3 |
| 8轮 MTP=off 第7轮/99行 | prefill_ms | 24.166 / 37.183 = 65.0% | 22.165–24.288 | 36.389–38.321 | 21.623 (n=3) | 3/3 |
| 8轮 MTP=off 第7轮/99行 | decode_tps | 94.647 / 90.857 = 104.2% | 94.646–96.057 | 90.725–91.468 | 92.386 (n=3) | 3/3 |
| 8轮 MTP=off 第8轮/107行 | prefill_ms | 24.631 / 37.644 = 65.4% | 21.441–27.821 | 36.238–40.181 | 21.496 (n=3) | 3/3 |
| 8轮 MTP=off 第8轮/107行 | decode_tps | 94.212 / 91.415 = 103.1%† | 79.379–96.452 | 90.873–91.700 | 92.436 (n=3) | 3/3 |
| MTP=off | parent_ready_s | 2.165 / 1.963 = 110.2% | 2.114–2.265 | 1.920–1.975 | — | 3/3 |
| MTP=off | gpu_mib | 4595.078 / 4263.176 = 107.8% | 4595.078–4595.078 | 4263.176–4263.176 | 4297.156 (n=3) | 3/3 |
| MTP=off | private_mib | 7062.363 / 5813.582 = 121.5% | 7061.109–7063.004 | 5813.121–5816.098 | 5511.430 (n=3) | 3/3 |
| 8轮 MTP=on 合并 | prefill_ms | 198.396 / 448.184 = 44.3% | 194.927–201.984 | 437.960–452.032 | 205.441 (n=3) | 3/3 |
| 8轮 MTP=on 合并 | decode_tps | 120.131 / 105.368 = 114.0% | 119.525–120.533 | 103.674–105.526 | 109.261 (n=3) | 3/3 |
| 8轮 MTP=on 第1轮/105行 | prefill_ms | 24.857 / 121.388 = 20.5% | 23.835–26.759 | 118.314–133.041 | 34.282 (n=3) | 3/3 |
| 8轮 MTP=on 第1轮/105行 | decode_tps | 114.805 / 92.452 = 124.2% | 114.494–115.465 | 92.163–93.266 | 108.404 (n=3) | 3/3 |
| 8轮 MTP=on 第2轮/108行 | prefill_ms | 24.247 / 45.929 = 52.8% | 22.480–24.562 | 45.770–49.459 | 24.050 (n=3) | 3/3 |
| 8轮 MTP=on 第2轮/108行 | decode_tps | 120.212 / 107.940 = 111.4% | 120.103–121.145 | 104.630–108.134 | 112.928 (n=3) | 3/3 |
| 8轮 MTP=on 第3轮/80行 | prefill_ms | 23.255 / 45.306 = 51.3% | 21.648–24.599 | 43.939–47.160 | 21.881 (n=3) | 3/3 |
| 8轮 MTP=on 第3轮/80行 | decode_tps | 114.191 / 80.442 = 142.0% | 112.160–114.749 | 76.296–81.385 | 98.453 (n=3) | 3/3 |
| 8轮 MTP=on 第4轮/114行 | prefill_ms | 24.717 / 43.851 = 56.4% | 24.248–26.071 | 43.068–43.924 | 22.627 (n=3) | 3/3 |
| 8轮 MTP=on 第4轮/114行 | decode_tps | 117.711 / 153.008 = 76.9% | 117.387–118.520 | 132.503–157.023 | 116.210 (n=3) | 3/3 |
| 8轮 MTP=on 第5轮/189行 | prefill_ms | 32.245 / 50.692 = 63.6% | 31.953–33.692 | 49.674–54.628 | 31.838 (n=3) | 3/3 |
| 8轮 MTP=on 第5轮/189行 | decode_tps | 126.342 / 98.738 = 128.0% | 126.101–127.061 | 96.901–100.210 | 115.659 (n=3) | 3/3 |
| 8轮 MTP=on 第6轮/110行 | prefill_ms | 22.065 / 43.709 = 50.5% | 22.019–23.103 | 43.017–44.022 | 24.210 (n=3) | 3/3 |
| 8轮 MTP=on 第6轮/110行 | decode_tps | 116.185 / 116.089 = 100.1%† | 115.974–116.979 | 114.829–118.782 | 103.833 (n=3) | 3/3 |
| 8轮 MTP=on 第7轮/99行 | prefill_ms | 23.006 / 44.081 = 52.2% | 22.077–23.738 | 43.027–50.286 | 23.529 (n=3) | 3/3 |
| 8轮 MTP=on 第7轮/99行 | decode_tps | 126.314 / 103.048 = 122.6% | 125.860–126.797 | 102.549–104.599 | 110.113 (n=3) | 3/3 |
| 8轮 MTP=on 第8轮/107行 | prefill_ms | 23.405 / 44.381 = 52.7% | 23.032–23.695 | 44.307–45.203 | 21.697 (n=3) | 3/3 |
| 8轮 MTP=on 第8轮/107行 | decode_tps | 125.852 / 121.257 = 103.8% | 125.733–126.396 | 117.737–121.359 | 114.672 (n=3) | 3/3 |
| MTP=on | parent_ready_s | 2.198 / 2.250 = 97.7%† | 2.193–2.240 | 2.237–2.403 | — | 3/3 |
| MTP=on | gpu_mib | 5069.082 / 4575.309 = 110.8% | 5069.082–5069.082 | 4575.309–4575.309 | 4639.289 (n=3) | 3/3 |
| MTP=on | private_mib | 8000.930 / 6670.121 = 120.0% | 7996.766–8005.457 | 6668.941–6670.781 | 6363.883 (n=3) | 3/3 |


A使用用户指定的两条stock命令：`-ngl99 -r5 -p512 -n0 -d0,4096,16384,30720`、`-ngl99 -r5 -p0 -n128 -d512,4608,16896,31232`，另给`-fa1/0`和`-o json`。llama-bench的tg128随机喂token、不采样；Hyle生成128个输出，首个来自prefill，127个decode步骤含greedy采样。非零深度时，Hyle先处理d行并生成1个待提交token，再输入511个新ID加该token，严格计时512计算行。已核对实际context_before和行数。

Hyle全程C32768；stock llama-bench请求context=d+p+n，内部可以对齐取整，未改其源码强制32K。B/C/D双方均为C32768，上游日志实际`n_ctx_slot=32768`；Hyle冷规划`future_shape_hints=0`。A前缀来自已有真实长文本token文件，B/C使用已有8个不同prompt，每轮64个输出；每个引擎以自己的输出接续，分叉后的历史内容可能不同。上游使用原生`/completion`、token数组、`cache_prompt=true`、`temperature=0`，Hyle沿用原入口；两者服务层工作不同。

启动统一比较进程启动至外部观察到ready，含DLL加载、权重加载和默认预热。Hyle内部ready约1.85s，未拿它直接对比server外部ready；上游以`/health`返回ok为准，轮询间隔10ms。第1轮请求里的延迟仍完整计入该轮prefill，没有扣掉或归给ready。每次新进程，未清操作系统文件缓存，不代表首次冷磁盘加载。

显存在独立重跑中由父进程的Windows PDH/PSAPI采样器测量，10ms目标间隔；实际时间戳保留在memory-samples.tsv，全部18个采样进程无setup/thread错误。计时进程没有该采样器。GPU峰是采样下界；私有提交是操作系统观测到的进程提交峰，不是RSS或物理内存，不与mmap工作集混用。

## MTP 接受 / 提案

Hyle默认K1，上游默认最多3个草稿。Hyle从导出的提交ledger逐轮汇总，并核对完整计数；不靠输出长度反推。

| 轮 | Hyle | 上游 | 冻结版 |
|---:|---|---|---|
| 1 | 25/37 ×3 | 34/85 ×3 | 25/37 ×3 |
| 2 | 25/37 ×3 | 37/75 ×3 | 26/36 ×3 |
| 3 | 23/40 ×3 | 28/99 ×3 | 21/42 ×3 |
| 4 | 26/36 ×3 | 45/54 ×3 | 27/35 ×3 |
| 5 | 27/35 ×3 | 34/83 ×3 | 27/36 ×3 |
| 6 | 24/38 ×3 | 39/71 ×3 | 23/39 ×3 |
| 7 | 27/36 ×3 | 36/80 ×3 | 25/37 ×3 |
| 8 | 27/36 ×3 | 39/69 ×3 | 27/36 ×3 |


## 后续profile方向

本卡不做优化。优先对齐标准pp512的512行形状、实际KV长度、命令提交和GEMM/状态算子成本；本卡没有采Nsight，不能从吞吐差直接判定主机或kernel根因。MTP第4轮应先对齐K1与上游默认最多3个草稿的验证工作量、接受分布和轮成本。普通启动的约202ms差额需按外部进程启动/DLL/引擎准备拆开。显存差额保留为总量，不在本卡追池字段。

## 复测

先在`.scratch`准备同tag的官方包及CUDA运行库，或在MSVC开发环境运行：

```powershell
git -c core.longpaths=true clone --depth 1 --branch b12063 --single-branch https://github.com/ggml-org/llama.cpp.git .scratch/llama-b12063-source
cmake -S .scratch/llama-b12063-source -B .scratch/llama-b12063-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake --build .scratch/llama-b12063-build --config Release --target llama-bench llama-server --parallel 6
```

本机完整参数（新测量换一个空输出目录，更新实际Hyle版本；仅离线重建本表可沿用window1）：

```powershell
$comparisonArgs = @(
  '--output', '.scratch/inference-upstream-head-to-head-20261103/window1',
  '--hyle', '.scratch/build-compute-v2-dev/bin/hyle.exe', '--hyle-version', 'd7c82a76f (inference core 5192b4232)',
  '--inputs', '.scratch/qwen35-unsloth-mtp-20261024/geometry-v2-check/original',
  '--model', '.scratch/architecture-refactor-2026-10-18/llama-matched-consumer/Qwen3.5-2B-text-BF16.gguf',
  '--mtp-model', '.scratch/architecture-refactor-2026-10-18/llama-matched-consumer/Qwen3.5-2B-text-MTP-BF16.gguf',
  '--reference', '.scratch/inference-prefill-close-20261101/long/peer-long1/peer_long.exe',
  '--reference-inputs', '.scratch/qwen-mtp-window-20261022/inputs',
  '--reference-version', 'd9e03f1074-ggml456172e-matched-consumer',
  '--long-ids', '.scratch/inference-long-context-20261101/workload/32736/prompt.ids',
  '--schedule', '.scratch/inference-mtp-gating-20261101/workload/real-eight-turn.txt',
  '--hyle-runtime', '.scratch/inference-default-product-window-20261031/peer-window/runtime-environment.json',
  '--reference-runtime', '.scratch/qwen-mtp-final-window-20261024/peer-link1/runtime-environment.json',
  '--official-bin', '.scratch/inference-upstream-head-to-head-20261103/official',
  '--source-bin', '.scratch/inference-upstream-head-to-head-20261103/build-b12063/bin',
  '--upstream-version', 'b12063/2cbca389f515df455412dafa07e2b1359cc72a6d',
  '--selected', 'official', '--fa', 'on', '--capacity', '32768'
)
# 在新目录依次运行prepare/select，读selection.txt后填写较快的selected/fa。
# 然后依次运行standard、multi、memory；GPU只运行当前进程。
python scripts/inference-upstream-compare.py validate @comparisonArgs
python scripts/inference-upstream-compare.py analyze @comparisonArgs
```

`prepare`只核对本地软件；`select/standard/multi/memory`实际运行模型，失败不覆盖原结果。服务默认localhost:2048并带`--metrics`。`analyze`只读日志重建表格，不再次推理。每个阶段完整命令、生产者原始输出和独立采样在[本次日志](../../../.scratch/inference-upstream-head-to-head-20261103/window1/)，[完整性检查](../../../.scratch/inference-upstream-head-to-head-20261103/window1/validation.txt)为54进程通过。

按轮、按阶段内存的常规工具仍为`python scripts/inference-benchmark.py --help`；Nsight范围采集仍用`inference-nsys-prepare.py`、`inference-nsys-capture.py`。本卡只更新最新上游数据，未改这些接口，也没有复做历史正确性/组件验收。
