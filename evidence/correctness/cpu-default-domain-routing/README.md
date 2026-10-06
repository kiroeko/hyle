# CPU 最终默认域规则与缓存身份资格

2026-10-06，Windows canonical 产品候选 `393c7545b06f599001a4066876f03595d0bb0b15`。
这是 CPU 收口的冷规则、默认消费和交付验收；GPU 暂停，没有新算子或新 P1。
[receipt.json](receipt.json) 和 [qualification.zip](qualification.zip) 保留实际命令、失败/纠正、
XML、274 场景消费者、资源探针、源码身份、选择器计划、预处理证明、LOC 与清理记录。
归档 1,448,118 bytes，SHA256 `316dca3a01a7a8c2da6bbf381e54a439d812fb5bbc96726c6c80b563f197bcce`。

## 实际结果

| 验收 | Windows MSVC 19.50.35728 | Linux GCC 14.2 |
|---|---|---|
| CPU facts/缓存/终止字符/并发读取 | 19/19 PASS，实际 Intel family6/model183/stepping1 | 19/19 PASS，同硬件身份 |
| Native factor，包括原数值、资源与 OOM | 144/144 PASS | 144/144 PASS |
| 普通 CPU 直接入口 | 13/13 PASS | 13/13 PASS |
| 普通 CPU Graph | 19/19 PASS | 19/19 PASS |
| Platform/Native 逐头 self-containment | 2/2 PASS | 2/2 PASS |
| 274 场景默认路由和两次改变输入后的 Direct/Graph 输出 | PASS；187 Native 优选、13 expert 冷拒绝 | PASS；0 Windows 性能标签，成熟默认绑定 |
| 9 类 epsilon/bias/broadcast/axis/scale/mask/shape/Fill/线程错配 | PASS，成熟回退 | PASS，成熟默认 |
| 同批成熟 operation+Copy region | 每次 152 个保留 | 每次 152 个保留 |
| 普通 Direct/Graph 独立并发槽与 close 后拒绝 | PASS | PASS |
| source-static、source-shared、installed-static、installed-shared | 四种实际完整旅程 PASS | 四种实际完整旅程 PASS |

消费者使用真实成熟 Provider 与 Native 数学，执行原 catalog 中除四个已删除 MatMul 外的 274 场景。
数值 control 使用两线程普通 Context 排除串行限定的 Native 优选，并断言成熟身份；这是数值控制，
不用于性能比较。旧 ND2/因子资格仍是数值边界依据，不以这批有限输入代替全部浮点语义。
Graph 使用实际普通 facade 与可复用 Invocation，未用 mock、枚举或 copy 冒充算子。

## 有限纠正和复用边界

- 首次消费者编译遇到 Operation 的 explicit 构造与非 façade 暴露的 ggml factory；改测试使用
  普通成熟 control Context，原失败日志保留。产品没有因该 fixture 错误改变 API。
- Fill 资格限原值 1.25，避免把旧比较外推到零或 NaN。9 类不匹配检查包含另一个 Fill 值。
- 红队发现初候选 `7090222a` 的编译条件缺少 NDEBUG。依法退回 AUDITED，只增加该冷编译条件。
  实际 MSVC 预处理证明 Release 资格函数体完全一致（摘要见 build-mode-proof.json），
  非 NDEBUG 分支直接 false；没有宣称跑过完整 Debug SDK 或其它 MSVC 版本。
  旧 owning XML 按 Release 函数/算法/布局不变复用，最终候选的 Windows 四种实际旅程再次通过。
- Linux 原共享安装失败来自未纳入计算 target 的旧 platform_thread.so，其 RUNPATH 不符本次
  install 脚本。补构建已选 SDK 闭包，只重链接该未变实现，安装和共享消费者通过；未改产品或
  公共工具。原失败保留。初始 Linux 日志的 static-source-ordinary 名称沿旧驱动，实际初始
  linkage 为 shared；随后显式 HYLE_STATIC=ON/OFF 的两种源码和安装旅程各自实际通过。
- 冻结 ND3 比较仍在原证据；187 个收益仅用于相同数学/参数和已测硬件软件条件的开发期规则。
  不把当前候选签为新 P1，不签全 Provider/全 CPU/全模型 SOTA；原失格、LOSS、微小 WIN 不改名。

## 保留源码的真实依据

每族有明确共享 WIN、已验 whole-domain 或资源消费，见
[域账本](../../../docs/implementation/tasks/compute/cpu-default-domain-ledger.md)。
SDPA rows1 不是性能 WIN；资源探针在 Windows installed-shared、Linux installed-static/shared
实际证明 Native private invocation=0、成熟私有上界未知，普通 Graph 在 invocation<=1MiB 下
选择 Native，Direct/Graph 输出实际一致。不会以默认后备优先级本身作为保留理由。
13 项已测劣势 exact 域在既定资格条件下仍明确拒绝，不因资源探针复活 rows32 SDPA。

共同 Native 数学、SIMD、原 ATen/TorchAO/ds4、ggml 与 oneDNN source 相对本轮 baseline
逐文件一致；只改冷 Context/propose/prepare/factory 和路由元数据。原 NativePlan/Invocation layout、
submit、allocator/资源/完成程序没有改变。whole-domain 分支明确不参与 exact 优选/退役；原资格复用。

## 选择器和适用风险

原保守 selector 计划及 skipped reasons 原样保存，产品授权为 READY。新增 CpuIdentity 仅追加 API，
既有 public signatures/layouts 与调用路径未变；只有 Native 冷 factory 实际消费新函数。
owner CPU tests、两个 header suite、真实 CPU Direct/Graph 和三交付覆盖新增风险。
其它传递闭包的 IO/Container/Process/Sched/Simd、GPU 等没有实际执行语义/ABI 变化，不新增测试根，
不把未运行门写 PASS。SIMD 源码保持，未签新的 SIMD 性能；GPU 算法与验收未重开。
P0：固定表有限匹配、只读 metadata、硬件缓存、无用户 payload 读取；热提交无搜索/计时/裁决。

## LOC 与临时目录

统一 `scripts/loc-report.py` 对不可变候选的 29 个 Compute+Graph 路径统计：
**61,109 实现行 / 249 文件**，不含测试、bench、第三方与 scratch；相对本轮 baseline 增加
414 行冷路由/规则，新增数学算子为 0。先前 MatMul 源级退役删除 691 实现行及两个文件不恢复。
两平台四个本轮消费者 build 目录已检查路径、CMake owner 与链接后删除，释放
**678,437,042 logical bytes（约 647 MiB）**；冻结性能 SDK、可复用 producer、依赖 cache 与本地模型保留。
