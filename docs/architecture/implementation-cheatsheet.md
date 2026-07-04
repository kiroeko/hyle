# 实现速查表（Implementation Cheatsheet）

> **每次写代码前只读这一张。** 它聚合各设计文档里"每次实现都要用的硬规则",降查询税。深层 nuance 仍在各 design 文档,**只在动那个模块时按需深读**(写 sched 才读 sched-design,等等)。
>
> 本表只抽取**铁律 + 判据 + 流程**,不重述设计推理。冲突时以被指向的源文档为准。

---

## 0. 北极星(一切取舍的标尺)
- **核心目标 = 性能 SOTA**。标尺不是 std(赢它是最低门槛),是各领域**参考库**(hashmap→absl/F14、分配器→mimalloc、队列→tokio/moodycamel、SIMD→Highway)。**做到等同或超过才算数。**
- **服务对象(双北极星,都四样全压,合起来压满五支柱)**:① **类 llama.cpp 推理引擎**(io serving+HTTP+批调度+KV-cache+GPU/CPU kernel,拉满 io/sched/mem/compute) ② **CAD/BIM/CAE 工程计算管线**(参数化/仿真依赖 DAG 增量重算+几何/BVH kernel+**CAE 稀疏线代/迭代求解器**+大场景图·网格·矩阵+云端协同/HPC,额外拉满 **graph** 支柱——clay taskgraph 老本行,CAE 再把 compute/simd 压到求解器深度);③ 蜂群 agent(消费者之一,压 fiber+io+channel)。热路径 = fiber/io/channel/mem/hashmap/SIMD/**graph 重算**。

## 1. std 使用纪律(分层,源:coding-conventions §1)
- **公开头/跨 DLL/C-ABI 签名禁 std 拥有型**:`std::string`/`vector`/`unordered_map`/`shared_ptr`/`function`。用 `string_ref`(视图)/`CArray`/SwissTable/侵入式 `Ref<T>`/函数指针。**硬线(可 lint)**:公开头 `#include <string>/<vector>/<unordered_map>` 用于导出符号 = 违规。
- **全库禁 std 流**:`<iostream>`/`cout`/`cerr`/`endl`/`<sstream>`/`ostream`。诊断走 `std::fprintf(stderr,...)` / obs log;格式化用 `std::format`(值语义)不用 `<<`。
- **生产禁 std 垃圾容器/哈希**:`unordered_map/set`/`map/set`/`std::hash` → 用 SwissTable / XXH3 `Hash<T>` / `CArray`/`SmallVector` / `string_ref`/`string_buf`。
- **内部实现可自由用 std**(非导出、非热路径);`std::expected` 是钦定错误模型。
- **std 取代是否伤 SOTA?已逐条裁决**(coverage-map ①′):intrinsic 背书(`<bit>`→popcnt 等,objdump 证)/编译期零运行时/已实测赢/冷路径——无一落热路径。**新用 std 顶替热路径件前,先查该裁决或实测。**

## 2. 失败模型(源:lifecycle-design §0 / charter)
- **自家 bug 绝不 catch** → fail-fast:`HYLE_FATAL_ASSERT`/abort + 诊断。**默认 no-throw**;破例须"不伤热路径 + 为隔壁稳定 + catch 外来非自家 bug"且**拿数据撑**。
- 运营失败显式返 `Status`/`std::expected` + `ASSIGN_OR_RAISE`。catch 只在 adapter 边界做"外来异常→Status",绝不外泄(异常跨 C-ABI = UB)。

## 3. 分层判据(源:roadmap 依赖图 / scope)
- 依赖图无环:`error→platform→mem→container→sched→io/compute→graph`。**高层依赖低层;反向 = 架构倒置(禁)**。
- 唯一例外:container(T0) 内联自包含 SIMD 探测片段,不依赖 simd(T1)。
- **若实现 X 必须依赖更高层的 Y → 路线图/分层有问题,先修分层,别半做**(用户铁律)。

## 4. C-ABI 边界(源:conventions §4)
- 对外纯 `extern "C"` C 头 shim(ggml 式)+ 内部 C++23。
- `make_X()/destroy_X()` 配对,分配责任锁 DLL 内(防 CRT 堆不匹配),禁客户端 `new/delete`。
- 跨 DLL 方法标 `HYLE_<MOD>_API`(类有非 POD 成员时宏放方法不放类,防 C4251)。

## 5. 调用安全边界(源:usage-contract.md)
- **fiber 体勿跨 `yield()`/`suspend()` 持有 thread_local 派生值**(work-stealing 迁移 + -O3 缓存 %fs → 读错线程)。yield 后重新访问 / 用 per-fiber 状态。
- 分配器跨线程 free 安全;**勿 static+LTO 内联 mi_* 进 fiber 体**(打破不透明边界)。
- **CPUID 探测代码必须 baseline ISA + 独立 TU + 禁 LTO**(否则老 CPU 探测自身 SIGILL)。
- 能力 dispatch:组合优于包含——加厂商特化 = 加 1 组合单元、改 1 文件、0 调用点改、0 热路径开销。

## 6. 里程碑闭环 CHECKLIST(机械跑,别靠自觉)
1. **真做**:每步 Write/Edit/Bash → 看到真实结果再下一步;绝不在思考里"叙述"成已完成。报后台结果前先 `cat` 其 output。
2. **吸纳 clay(能力查漏)**:闭环前对照 clay 同名文件(coverage-map)查有没有漏掉仍需要的**能力**,缺的补或显式登记。目标=能力覆盖+性能≥clay,**接口不必一致**(不追平 clay 签名/命名)。
3. **真编真测**:本轮真发构建、看输出无 error(stale exe 假 PASS);**双平台** Win MSVC + Linux GCC14;lock-free 必过 WSL TSan(`setarch -R`)。
4. **对抗自审**:闭环后挑刺——正确性/性能偏离/规范偏离/对照 clay 覆盖;发现就修或登记。
5. **cloc 验真**:量该里程碑增长写回阶段文档 + 索引;远低预期要解释(精瘦 vs stub)。
6. **提交三连**:`git status --short` → commit 只写真实出现的文件 → 核对 "N files changed" == 预期。
- **铁证信号**(任一不符=立即核对):测试数没变(14→14);commit files < git add 数。

## 7. 构建/环境(源:reference 记忆)
> 从零建环境的完整步骤(工具链门槛 / vcpkg+OpenSSL / 各命令)见 [`docs/dev-setup.md`](../dev-setup.md);这里只留每天用的速查。
- **Windows MSVC**:vcvars18(`Studio/18/Enterprise/...`)+ Ninja + Release(-O3);git-bash 调 `cmd //c` 易出引号/编码坑,**用 Python 写完整 bat 再跑**最稳。
- **Linux/WSL**:`CXX=g++-14 CC=gcc-14`(默认 gcc-13 过不了 CMake 门);独立 build 目录。
- **⚠️9p 铁律**:WSL 下一切**重 IO/压力测试**必须在**原生 ext4**(`git clone`/`rsync` 到 `/root/hyle`),**别在 `/mnt/c`(9p,不稳/CPU 不饱和)**。
- mimalloc 强制常开(无 off-switch);TSan 用 `-DHYLE_SANITIZER=tsan` + `setarch -R`。
- **TLS(io-08)**:可选依赖,Linux `apt install libssl-dev` 零配置;Windows vcpkg `openssl:x64-windows` + `-DOPENSSL_ROOT_DIR=C:/public/vcpkg/installed/x64-windows`。

---

> **写代码前**:过 §1 红线 / §3 分层 / §5 调用边界。**写完后**:走 §6 闭环。**对标**:§0 参考库实测,不只赢 std。
