# 已准入native资源公开默认构造修正

状态：CLOSED，2026-10-13；owners `compute_route_apple`、`compute_route_vulkan`，baseline
`1ff3bffba`。归原G10/交付接口缺口，不新增G项；CPU/Apple/Vulkan已签NR数学/性能不重跑。
真实trigger为[原比较器编译失败](../../../../evidence/correctness/inference-g01-native-ragged-p1-preflight/README.md)：
Apple MemoryResource inline default ctor包含unique_ptr<incomplete MemoryState>，消费TU
默认构造会实例化私有删除器而编译失败。Vulkan同公开形状；公开资源的空状态/随后
create迁入是自然普通旅程，不能要求消费方改用initializer来掩盖已承诺入口不完整。

主consumer是普通public C++默认空owner→create结果move assignment→释放，反偏置是
std::vector资源owner默认resize与move/destroy；不依赖bench/private产品头。有限FIX
MC-01：两公开默认构造声明标各自原export macro，定义搬到原MemoryState完整的TU
并=default，保原noexcept/空状态/大小/成员/allocator/失败/借用合同，不加兼容wrapper。
MC-02：source与实际relocated static/shared public-only consumer，单独include每个
公开memory头后default构造，旧prefix保留；empty/default、factory move、vector resize
和最后Context关闭。没有数学/copy/shader/Queue或新factor/依赖改动。

预算为短接口审计/实现与一次目标交付，按用户常设授权延长并保原失败。P0：空对象
构造不分配、不新增常态字段，普通已有factory路径不变，不跑P1/全部算子。Windows
仅Vulkan目标四jobs编译与同public consumer编译；实际MacBook Air身份先核验，两平台
重型串行。san只空owner/move/destroy的适用宿主风险，复用原树；driver不作虚假声明。
public头/API seam由selector选真实owner/消费者，工具归因缺口独立保留，不改checker。
各owner必须SCOPED→AUDITED提交后才改四产品文件。

两路线实际public-only独立消费TU已复现同一incomplete-type compile FAIL，报错均从
默认空owner构造进入unique_ptr删除器；不是只从相似源码猜Vulkan缺陷。审计固定
MC-01/02，原private MemoryState完整定义在对应已有实现TU，无需新include/依赖。
原delete copy/move/noexcept/ref/invalid empty合同全部保；默认构造无allocation，只有
空unique_ptr初始化，外部调用需正确导出symbol。公开头只声明，私有类型不泄露。
source与SDKconsumer只include各自public memory头并resize owning vector；没有新增
模型策略、copy数学或执行器。旧NR terminal继续证明原性能，不把本构造修正当新P1。

不可变candidate `a3fa6be8cf22780491e8b383ef1aafb678db1e23`：仅两公开声明+两原TU
定义；API/算法/consumer固定，source两个实际小consumer通过后验四SDK/宿主风险。
P0无新增allocation或常态字段，原normal math/提交/资源方法未改，不重新P1。

## 七门与终审

| Gate | 实际证据与裁决 | Verdict |
|---|---|---|
| scope | 两路线public default构造实际red compile FAIL；仅MC-01/02、原noexcept/空状态与完整owner旅程 | PASS |
| architecture | 公开声明＋已有完整State的TU默认定义，共四产品文件；原布局/工厂/依赖/数学/Queue不改 | PASS |
| correctness | [证据](../../../../evidence/correctness/inference-native-memory-default-construction/README.md)：两source public-only default/vector/factory move/moved-from/destroy/Context close，六hostsan及实际changed TU/test flags | PASS |
| delivery | 四实际relocated static/shared同public consumer，只链hyle::compute，独立include每条memory头；Windows同facade编译＋链接通过 | PASS |
| usability_docs | 普通默认空owner不再需要工厂initializer绕过；空ref无效，copy删除、move/borrow与原guide规则保持 | PASS |
| red_team | 对比四产品文件无其他方法/成员改动；两平台相同缺陷分别复现，source/SDK正常和vector反偏置完整，不泄漏private头 | PASS |
| engineering_performance | P0空unique_ptr默认初始化，无allocation/新字段；只有冷对象构造符号归属修正，P1语义不适用，旧NR数学/性能证书按未变输入保留 | PASS |

第三方仍原锁/许可证；原标准unique_ptr所有权、create/move/destructor与HYLE resource
复用，无第二层资源owner、FFI/兼容wrapper。实际Air Mac16,12/M4/macOS27.0.1/Clang21。
原ASan detect_leaks=0限制保留，不签LSan/GPU math/shader/driver；该轮只验宿主owner。
Windows四jobs产品库已编译，首次consumer漏项目已有/wd4324对合法alignas padding的
warning策略，原FAIL保留；按原生成compiler/linker输入补该既有选项后1.484s编译与
0.047s链接PASS，无GPU run。原配置181.969s主要在IREE prepare_source验证副本阶段，
不是全库测试；不重配置同tree/不顺手改IREE工具。helper linker PATH解析失败保原记录，
改用实际MSVC绝对link.exe路径；没有改产品/编译器规则或放宽其他警告。

selector报告已归属且当前AUDITED，仍net product introducing-commit归因BLOCKED，
与既有工具记录（已删除的历史文件：`docs/implementation/tasks/build/validation-selector-shared-bench-paths.md`）同类；原计划保留，
实际授权顺序da440→a166→a3fa→a632与对应门直接可核，不改checker、不重开无改动模块。
docs/closure/diff检查均实际通过。旧NR有限终审继续独立引用
[Apple](inference-g01-apple-ragged-terminal.md)和[Vulkan](inference-g01-vulkan-ragged-terminal.md)，
本默认构造修正不取消其数学/成本证据，也不签整个G10或G01完成。

统一LOC@a3fa/self-check PASS：全库2,886,340，自实现不含测试175,079，两个route实现
目录联合13,107。有限MC-01/02及七门已裁决，两个owner本轮CLOSED，停止重复验证此
空构造。reopen仅可复现public构造/导出/生命周期/空状态错误，或真实consumer需原
合同未表达的机制；CPU/private pools、分页与原其余G项仍待原goal。
