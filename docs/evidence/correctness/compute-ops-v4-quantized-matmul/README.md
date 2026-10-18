# ComputeOps R2-Q 验证证据

产品候选 `50753347a0a18ba81c47fd623a882ca3d719af2b`；严格 Unix 测试修正候选
`210044ce0b8b4df0ab6b796e00331536d3f0263d`。二者 `include/`、`src/` 的 Git diff 为空；后者只调整
测试引用/花括号和施工记录。未包含 R2-D 未提交产品代码。

## 已执行

| 环境 / 入口 | 实际结果 |
|---|---|
| Windows MSVC 19.50.35728，原产品候选 selector | 14 门 PASS：13 个精确身份 receipt 复用，Graph 三交付门以 Ninja 两任务完成；82 项不适用/未选理由保留 |
| Windows 新 Q4 public consumer | source/shared/static 三形态通过；installed SDK 已 relocation |
| Windows 修正测试候选 | ComputeOps/Graph 单测及逐头 4/4 PASS；selector 只选择 docs/closure/configure 与这两个 factor，91 项跳过；SDK 未重建 |
| WSL Ubuntu 24.04 / GCC 14.2 | 同候选单测及逐头 4/4 PASS，Q4 source consumer PASS，统一 UBSan 单测/逐头 2/2 PASS |
| MacBook Air Mac16,12 / AppleClang 21 | 同候选单测及逐头 4/4 PASS，Q4 source consumer PASS，显式 deployment floor 14.4 |
| Mac mini Mac16,10 / AppleClang 21 | 同候选单测及逐头 4/4 PASS，Q4 source consumer PASS；只签 correctness/portability |
| W1-WIN 描述布局 | Operation 改动前后均 128 bytes / alignment 8；QuantizedMatMul 12 bytes / alignment 4，不扩大该工具链的 Operation |

测试包括 signed/affine Q4 完整端口、dtype/group/K/参数形状/物理格式拒绝、zero/overflow、不改失败输出、
跨 DSO custom/resource/Q4 描述、真实 GraphBuilder 发布与旧资源/alias 尾项。只证明描述与推导，未接入 Q4
kernel，不签任何路线已完成或新的性能赢家。P0 为冷 O(rank) 验证和已有执行路径源码检查；P1 不适用本次 schema。

## 原始材料与失败记录

- `windows-raw.zip` 保存 selector receipts、日志、Q4 consumer、修正回放及布局源码/输出；
  `selector-summary.json` 保留每门 verdict/reason/cache identity。
- 三个 `*-logs-210044ce.tar.gz` 保存各 Unix 主机日志；`unix-recipes.zip` 保存调用脚本、controller 日志及相同 consumer。
- 第一次 WSL Werror 的五处测试告警、手工 UBSan flags 与 Core 编译合同冲突、Air embedding 缺少
  deployment floor 的失败日志均保留。产品修正只有一次测试引用/花括号调整；后两项通过既有
  HYLE_SANITIZER=ubsan 和明确父工程 deployment 参数解决，没有更改公共构建/验证工具。
- Windows Visual Studio 的嵌套 /MP 超出两编译进程，已停止其未完成 Graph 门；改为 Ninja 后确认
  两个编译进程。中止尝试不算 PASS，也未用作性能证据。所有后续重型验证串行。

原候选完整 Git archive SHA-256：
`adf23857d757bdd5b9bc9471306f3a01b093d0dd1b4a077abf5c2e8377b6dfcb`。
修正候选完整 archive SHA-256（WSL/两 Mac 已核对）：
`0f01e602d80908644e59e5e254db246c755c19d5d02a12d64268279e369589cf`。

`loc-210044ce.json` 由统一脚本生成且自检通过：Compute+Graph 实现 **31,220**，全库自实现不含测试
**126,405**，全库代码 **691,833**。LOC 包含待迁移旧代码，不是完成度。压缩材料校验见 `sha256.json`。
