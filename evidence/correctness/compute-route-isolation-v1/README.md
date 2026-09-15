# R1 路线构建与交付隔离证据

候选 `37dc8b542d52361cfa7b6cc176dc65555941e48d`，tree
`baf398dcf8aa42d5968a3cebe297133c3efc7b95`。产品 baseline 为 `ec142a92`，AUDITED 授权 baseline
为 `e469747ce768d352e4a9ac6d74f11edcb354f49d`。本记录只证明 R1 的选路与交付；R2–R7 的自然新接口、
原生计算、默认规则与性能没有从旧 runtime 或这些 smoke 获得新资格。

Windows 为唯一产品来源。WSL 和 MacBook Air 使用该候选的 Git archive 验证副本，archive SHA256：
`79975f409373656a6747d8ce7b98ba3a465d6aaca68c706122cf363e1d043836`。
所有重型构建串行，最多两 jobs；没有 GitHub Actions 或 P1。

## 结果与原始记录

| 范围 | 结果 | 证据 |
|---|---|---|
| Windows selector | PASS：92 selected gates，5 skipped | [summary](selector-summary-37dc8b54.json)、[完整 receipts/logs](selector-evidence-37dc8b54.zip) |
| Windows 具名交付 | PASS：19 cases，含六个独立 staging v2 cases | [summary](windows-r1-summary-37dc8b54.json)、[原始日志/命令](windows-r1-evidence-37dc8b54.zip) |
| WSL 具名交付 | PASS：12 cases，静态 CPU shared-factory 使用明确 PIC producer profile | [summary](linux-r1-summary-37dc8b54.json)、[原始日志/命令/失败尝试](linux-r1-evidence-37dc8b54.zip) |
| MacBook Air 具名交付 | PASS：7 cases，CPU/no-backend 与三交付 custom consumer | [summary](macos-r1-summary-37dc8b54.json)、[原始日志/命令](macos-r1-evidence-37dc8b54.zip) |

每份 summary 保存原始 log 与 ZIP 的 SHA256。selector ZIP 保留未裁剪的 selector result、receipt 与
对应 raw logs；compact summary 只提取 verdict/选择理由，不伪造另一份 receipt。

Windows：MSVC 19.50.35728/toolset 14.50.35717，默认 selector delivery 使用 Visual Studio；显式
路线和 custom consumers 使用 Ninja。WSL：GCC 14.2、CMake 3.28、原生 ext4、Linux CUDA 13.3.73。
Air：Mac16,12/arm64，Apple clang，Ninja，三种交付统一 deployment 14.4；实际工具版本在日志中。
Mac mini 未作为本卡要求的验证机，Apple GPU/Metal 路线资格属于 R5。

### 已执行的旅程

- CPU-only、无 backend、CUDA-only 各覆盖 Compute/Graph，逐一验证源码嵌入、installed shared、
  installed static 和 relocation。Air 只执行 CPU-only 与无 backend。
- CPU OFF 排除 CPU backend/Native/CPU oneDNN 的公开交付；CUDA OFF 不发现 Toolkit；缺席实现不
  进入 installed headers/targets/features。源码 checkout 仍含完整源文件，能力由真实 target 决定。
- Windows component-contract 覆盖 CPU/test 开关组合、CPU OFF + oneDNN ON 拒绝、AUTO/OFF 不探测、
  无关 root 和 test dependency 隔离。SDK smoke 检查真实 headers、targets、features 与依赖投影。
- 既有 CPU/oneDNN/Graph 和真实 CUDA backend/late-failure tests 已由 selector 回放。只按被选中的
  header suite 和交付 gate 取证；没有独立 header suite 的 io_core/simd 保留原 skipped reason。
- 自定义 CPU/CUDA 计算通过 source/static/shared 消费；CUDA 包含实际 kernel、变化输入、capture、
  多 lane 依赖与接纳后错误的 Completion。CUDA package fill/readback 仅为资源交付证明，不代表 R4 完成。
- Visual Studio/Ninja 各验证 source/static-install/shared-install staging：首建、DLL-only 增量、无
  改动、EXE/import SHA 不变、空集合与重复调用。真实 Compute/Graph VS 三交付也通过。

## 失败、纠正和交付条件

1. 初始 selector 被 ggml 空审计身份误分类阻塞，由独立 VI1 工具卡修复；ggml 原 BLOCKED 与恢复
   条件不变。工具回归 45 项通过，不把无产品身份写成已实现 Provider。
2. `8158d74d` 的 Visual Studio 源码首建在 Native DLL 生成前 staging，独立 v2 修复后由候选重新
   取证。旧失败 log 保存在 Windows ZIP；不是靠第二次 build 或手动复制通过。
3. WSL 的首次 CUDA configure 从继承 PATH 选择了 Windows `nvcc.exe`；Linux SDK 与 cudart 已安装。
   修正任务环境的 native CUDA PATH 后，用新 `-native-sdk` 目录仅重跑 CUDA 两组。原失败目录保留在
   验证工作区，日志和 SDK 选择 cache 保存在 WSL ZIP，CPU/no-backend 通过记录没有重跑或改名。
4. Linux 普通静态 SDK 已通过 EXE 消费；把非 PIC archive 链进 CPU custom shared factory 时失败。
   生产静态 SDK 时使用现有 `CMAKE_POSITION_INDEPENDENT_CODE=ON`，新 SDK build/install/relocation 和
   原失败 consumer 通过。原 negative 及 PIC producer log 均保留；没有改产品默认 flags 或编造普通
   non-PIC archive 已能装入 DSO。此条件在 distribution 和 sample 指南说明。
5. Visual Studio staging helper 在创建 consumer target 的同一 CMake 目录调用；跨目录库依赖
   仍由实际回归覆盖。用户指南说明该调用约束。

## P0 与规模

相对 `ec142a92` 的产品 C/C++/CUDA/Objective-C 源码和公共头 diff 为空，新 CPU capability 宏没有
库 runtime 使用点。没有算法、对象布局、常态 metadata/dispatch 或资源算法改动。额外 PIC 验证是
既有 producer 编译配置的消费条件，不签不同编译配置之间的性能等价。P1 对本轮 build/install 改动
不适用；没有新增性能胜负结论。

[统一 LOC 原始结果](loc-37dc8b54.json) 自检通过：Compute+Graph 全部实现 **31,135** 行，不含测试；
全库自实现不含测试 **126,320**，全库代码 **689,555**。包含全部现有 backend/provider 和 compute_ops，
也包含待迁移的旧代码；LOC 不是本轮完成度。

最终 docs/closure、状态迁移、七门和 reopen 条件由各 owning task 记录。本 evidence 不单独授权产品修改。
