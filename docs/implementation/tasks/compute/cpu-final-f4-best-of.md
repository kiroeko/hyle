# F4 CPU全重复域批量best-of

<!-- MODULE REVIEW: SCOPED -->

当前包SCOPED，baseline `83240ab43ba80cd549dcf03933e3d69e8f94c2b8`（登记完整revision），F1/F2/F3已完成。
owner compute_provider_native、compute_route_cpu、graph_route_cpu；总合同[CPU最终收口](cpu-final-closure-plan.md)。输入F2重叠域与F3当前合格承载。
只比较实际可执行且同语义/精度/资源/完成边界的候选，覆盖oneDNN/ggml/ATen及剩余Native，
完整region与等价完整程序比较，保留融合。单承载只正确性/资源/P0，能力差异不能算loss。
按真实模型角色、原格式、dtype、布局、attrs、shape、线程与软件/硬件身份归并，normal/holdout预冻结。
API/算法稳定后每批一个不可变候选，原仪器四strata/AA/controls/固定n15与5%实用门，重型串行。
一轮有限preflight/适用P1与至多一批具名接线纠正；不自研优化、不n31寻找PASS，未知不得解释平局。
按用户最新效率取舍，接近或预算内无法裁决时快速选择oneDNN/ggml等完整成熟路线，记录
`ENGINEERING_DEFAULT`及格式/融合/转换/资源/交付依据。无需重复长资格采样，不假写赢家；
该项以正确接入/资源/P0和明确性能限制验收，未裁决测量保持原状态。只有具名material风险仍需原适用门。
稳定方向与实用门分别记录；赢家按低成本冷规则绑定，submit不搜、不计时。旧187/17不自动取得全候选最优。
先完成固定新版oneDNN，不能测完旧版再升级；自研同合同落后且有完整成熟接替即交F5删除，优化挽留预算0。
规则显式覆盖已验平台/版本/硬件/线程/真实模型端口；未知身份走合法成熟基线，不承诺所有机器绝对SOTA。
通用默认必须逐合同归属，而非沿oneDNN→ggml→ATen全局顺序；原压缩矩阵/完整融合/稳定函数等分别确定
合格成熟承载与依据，复用库内ISA，精度/错误/资源不匹配明确排除。验证特化命中、未知条件合理回退及feature裁剪。
数值/资源/交付缺口的MUST仍是阻塞；仅性能无法区分的成熟候选按用户授权工程默认裁决。
不得宣称全部规则均为统计性能WIN；各批规则正确性/资源/三交付同时完成。

## F1等待期间的有界只读输入对账

基线 `36123059f23e409978499681ee2df748e74d386f`；本节不启动F4产品实施、不改变registry owner，
也没有运行P1。F1 Linux混合loader仍待，完成后再按协议绑定SCOPED/AUDITED与有限FIX。

已读 `bench/compute_provider_native/installed_p1/native_domain_bench.cc` 和其两份contract，
与当前三个成熟factory、F2端口矩阵及Native源码对账，确认以下有限迁移需求：

| 旧验收入口的具体限制 | F4必须固定的迁移 |
|---|---|
| Session只以bool选择Native或普通CPU默认栈 | 比较具名oneDNN/ggml/ATen与实际self候选，不把默认栈替代全部候选 |
| catalog仍含Native MatMul和SDPA，当前两项已物理退役且原plan明确拒绝 | 这两族继续比较实际成熟交集；从self裁决集删除已退役候选，不复活源码 |
| contract固定旧oneDNN3.13.2/旧ggml456与历史206/72预资格结果 | 按当前不可变SDK/实际软件和新的预资格批绑定；不继承历史WIN或AA结果 |
| 通用宽度1536与rows1/32不能代表F2全部真实角色 | 从固定需求选有限normal/holdout；完整内宽/编码/precision/资源不偷换，未测域成熟工程默认 |
| Native identity同时交付自研与借入原程序 | 逐源码族区分self与成熟codec/程序，不能按Provider名字整项删借入能力 |

普通用户入口继续只选CPU路线；上述显式候选控制仅属于开发验收。新比较保留完整准备/重放、
转换/packing/临时与完成成本，Graph完整region单独同合同对账，不把拆叶子结果外推融合。
复用现Data/Session、统一仪器/controls/四strata和固定n15，只迁移具体benchmark consumer与contract，
不重写公共性能核心；具名新接线缺陷最多一批纠正，近似或噪声按既定工程默认快速裁决。

## 本轮SCOPED：固定owner与有限审计

真实trigger：F1固定oneDNN3.13.4与单/混合CPU core交付后，旧187/17证书不适用；普通模型CPU两入口
需要当前真实同合同候选裁决、未知条件合理默认，并向F5提供源码族去留依据。旧harness的具名限制见上节。
Native与CPU route原F3 CLOSED、Graph CPU route原F1 CLOSED按此新版本/完整候选消费要求重开；
成熟oneDNN/ggml承载owner保留原CLOSED，只有具体接线反例才另按协议定向绑定，不能泛化重审。

| 有限审计 | 固定输出/允许后续FIX边界 |
|---|---|
| A1 实际重复合同 | F2→当前源码的dtype/precision/layout/shape/attrs/资源/状态交集与完整region；单承载不跑P1，self与借入程序分清 |
| A2 批量consumer | 复用原Data/Session与仪器，改具名候选/当前SDK身份与有限normal/holdout，不复活退役MatMul/SDPA，不写新数学 |
| A3 准备与执行账单 | 准备、转换、packing、临时、线程、checks、同步/完成与融合损失同边界；资格差异排除，不计LOSS |
| A4 有预算裁决 | 一批preflight，固定四strata/AA/controls/n15/5%；明确差距与方向分别记录，接近/噪声快速成熟工程默认，不追加长循环 |
| A5 普通规则 | 既有冷规则/private表按真实格式/模型/硬件/软件/线程/recipe绑定；完整region优先，未知合法默认、无在线计时、submit不搜索 |
| A6 退役交接 | 35种/15源码族逐项映射；没有完整优势且成熟完整替代的self交F5物理删除，独有资源域明确，SIMD/Mem基础不删 |

主consumer：固定目标模型的projection/norm/index/state/完整片段普通Direct与Graph；正交consumer为
无模型身份的tensor批处理及相同操作的不同真实role/布局/holdout，不由旧benchmark宽度定义API。
公开API/layout、Provider资源与原执行程序不变；SCOPED只授权有界审计，AUDITED有限FIX冻结前不改产品。
current source/static/shared与三平台资格沿F1/F3对账，规则新命中/失配及新默认在本包完成两入口/资源/三交付验证。
无新依赖升级、无自研优化、无模型loader或GPU资格队列；F4结束前不启动F5删除。
