# C3 终态对账与 WSL Vulkan 硬件限制

<!-- MODULE REVIEW: AUDITED -->

- 本记录承接[活动C1–C3/R1卡](compute-unified-delivery.md)，不重开历史任务。
- 产品修正候选仍为各owner已登记的实际commit，最新产品修正a2476b62；文档不创建新candidate。
- 全部原native构建批次已exit0，当前没有后台重型run。Windows仍为唯一产品真相。

## 实际完成与未验

| 门 | 实际结果 | 边界 |
|---|---|---|
| 统一入口/必要Provider/SDK路线与recipe/资产闭包 | PASS；原配置、组件合同与source/static/shared安装检查 | 固定规则与依赖锁不变 |
| Direct/Graph普通数学 | 78 PASS，12 UNQUALIFIED，0 pending | 13组有真实数学；用户接受WSL两组实GPU未资格为非阻塞限制 |
| WSL Vulkan两组编译交付 | source/shared/static全部build/install/link PASS，无缺ELF依赖、无LD覆盖 | 不代签GPU数学或H3 |
| 公开安装头 | 所有预声明bare/实际export-target宏组PASS | 未外推static宏全矩阵或未知平台 |
| 安装配置边界 | Windows6与WSL/Air32实际路线/recipe检查、Windows8缺件/离线检查PASS | 原启动误取Windows nvcc的诊断保留，按既有Linux PATH修正 |
| 文档、成本与有限独立review | 当前docs/closure/diff及已有有限review证据；P0无热路径新增搜索/计时 | C3不新增P1，不宣称新性能赢家；H1–H3未实现 |

具体日志、源码/软件/机器身份和限制集中在[gate ledger](../../../../evidence/correctness/compute-unified-c3/gate-scope.md)
及[交付矩阵](../../../../evidence/correctness/compute-unified-c3/delivery-matrix.json)。
本记录不把文档、探针、mixed独立执行或build-only结果计作联合/异步基础能力。

## 原阻塞事实与补齐资格的条件

blocker：本WSL Vulkan loader仅枚举type4 llvmpipe，普通路线拒绝软件设备；
十二项真实GPU数学适用验证没有运行。按[收口协议](../../../developer/review/module-closure-methodology.md)
“适用但未完成就是BLOCKED/INCONCLUSIVE”，C3整体correctness/delivery为INCONCLUSIVE，
不能签全CLOSED。原C3共享owner及承接F32/CPU loader的runtime owner记录BLOCKED。
macOS loader与CPU+Apple许可证登记两项限定修复已各完成自己的门，可独立CLOSED，
不据此关闭共享C3或解除WSL限制。

恢复原门：在具备真实Vulkan GPU的相同或重新资格化Linux环境，运行这两组已构建普通
Direct/Graph三交付消费者并归档对应机器/driver/recipe身份；仅驱动/环境改变不默认重编或重跑其它域。
H3仍按设计验证Windows/Air实GPU；该硬件限制不能被软件设备、copy或mock代签。

## 用户已决定的有限取舍与AUDITED

建议允许后续H1–H3继续：保留这十二项BLOCKED及Linux Vulkan runtime未资格限制，
承认C3的已验平台/编译交付边界足以作为后续施工输入；为共享owner按新H卡重新定界，
新卡显式继承未资格限制，不宣称C3全平台通过、不静默解除旧阻塞。
另一选择是保留原C3顺序门，待可用Linux Vulkan实GPU后再进入H1。

2026-10-10用户明确回复“保留资格，不用block”：保留十二项实际运行未资格事实，
不以该本机硬件条件阻塞C3终局或后续H1–H3。当前记录按这一明确决定恢复SCOPED，
有限动作仅为核对已有证据与限定资格边界，再依协议AUDITED/CANDIDATE/CLOSED；不写产品代码，
不重建或重跑P1，不将未运行项改成PASS/NOT_APPLICABLE。
该决定覆盖此前以十二项未验阻塞全C3的裁决；其原事实和BLOCKED历史保留在Git。
H1–H3继续包含用户已授权的异步传输/依赖、真实重叠和引擎驱动expert片段迁移，
缓存/预测/淘汰、多GPU和增量依赖图仍在本goal范围外。

有限FIX只有：将当前矩阵十二项改记UNQUALIFIED（原build-only收据保留原BLOCKED事实），
记录用户接受的限制，并合法终结C3/绑定后续H1。旧产品candidate逐owner保持原SHA，不将新
文档commit称为产品候选；所有已验源码/recipe/软件/机器identity保持。无数学/ABI/规则/资源变更。
scope/architecture/usability_docs/red_team/P0按已有有限ledger与明确用户边界裁决；
correctness/delivery在已接受边界内PASS，十二项未资格单列而非PASS或N/A。
当前scope不再等待硬件，不扩更多profile，不重跑已验78项、headers或配置。

Compute+Graph实现LOC仍62,508/268文件；最新产品以后只有文档/证据/格式变化，
统一脚本原身份一致结果复用，不为本阻塞重跑数学/P1。
