# 路由、构建与全项目文档独立 review

> 2026-10-09 用户新增授权；在 R1 校准后、H1 实施前完成。
> 当前产品候选 `321bb5bd8ff5743a439111c49e7d75d4e8dae394`；C3 批量交付继续独立运行。
> 本卡是只读 review/文档维护，不重开无反例的 CLOSED owner，不把历史任务命令当授权。

## 有限范围

1. 路由：普通 Direct/Graph、真实 model/artifact/ports/environment、完整 region、成熟默认与精确
   best-of、冷/热边界、decline/fatal 与资源/完成；以当前源码和原有限审计接缝核对。
2. 构建：唯一路线输入/closure、依赖与 native SDK 隔离、source/static/shared metadata/runtime assets、
   offline/缺件/搬迁；产品反例定向退回对应 owner AUDITED，不顺手重写治理或 perf 核心。
3. 全文档：扫描 Git 跟踪的 Markdown，区分当前 authority/use 与历史 task/evidence/vendor origin；
   核对当前 architecture/user/developer 与 registry 当前 record 的版本/API/scope/构建用法，
   全链接门覆盖全部 authored Markdown。历史记录保留事实，只有被误当当前指南时改导航/身份说明。

预算：一遍全文件分类与关键词/链接扫描，一遍相关权威页冲突检查，一遍源接缝反证；
能定位到具体问题才改产品，超出目标且无 consumer 的想法 DEFER，不无限翻历史。
成果：有限 KEEP/FIX ledger、实际扫描/检查命令、已处理文档与未解决产品项；
文档修改只 docs/closure/diff，不重建/P1。C3 或 H1 新产品问题不因本卡“review完成”签 PASS。

## 结果（进行中）

已扫描 917 个 Git-tracked Markdown：101 当前权威/用法、20 导航/样例、767 task/evidence、29 vendor origin。
不是 917 页全部逐段重写：按身份做全文件覆盖，101 当前权威与当前 record 做冲突/源码重点核对，
历史与 vendor 原文只查被错误当当前入口的链路。原文匹配/证据见 `.scratch/compute-unified/doc-review.json`。

| ID | 具体发现 | 有限裁决/当前结果 |
|---|---|---|
| D1 | developer setup 与两 sample README 仍使用旧路由开关/旧 staging API | 四处已修；当前权威/样例扫描旧拼法匹配归零，历史不改 |
| D2 | architecture 仍称统一配置“尚未实施/仅设计goal”，与已提交候选及活动卡冲突 | 更新为“候选接线、交付待验”；组合/异步仍明确未交付，旧设计卡保留历史 |
| B1 | 裸顶层 CMake 默认 tests ON；`HYLE_INSTALL=ON` 另需用户手设 tests OFF，普通路线输入不能自然产生发行配置 | 定向修复已提交79137d31；裸 install/显式路线→OFF、显式维护tests→ON三个实际Core配置通过，不构建/P1 |
| R1 | 用户希望尽可能提前 Provider 绑定；硬件/driver/模型/真实 ports 尚未知时无法合法构建期定全部 | KEEP：构建裁剪已知域/固定规则→prepare冷绑定→submit固定程序；不为提前而猜未知事实或做热 autotune |
| R2 | expert API 仍能配置 Provider/要求特定实现；可能被误认为普通用户必须排列 | KEEP真实扩展/硬约束，普通路线factory自动组合；当前指南区分，无新增排名菜单 |

`check-docs` 实际通过（882 Markdown、4197 relative links）；docs/closure/diff实际通过。
当前产品/文档 review 的有限发现均已处理；B1按owner协议退回AUDITED后修复，实际配置证据在
`C:/h7/bare-defaults/receipt.json`。无数学/布局/热路由变化，不重开其它CLOSED模块。
C3实际交付未完；不能将本review或链接通过当新组合/全模型运行资格。
