# 已准入native资源公开默认构造修正

状态：SCOPED，2026-10-13；owners `compute_route_apple`、`compute_route_vulkan`，baseline
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
