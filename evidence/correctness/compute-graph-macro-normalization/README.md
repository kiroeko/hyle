# Compute + Graph 宏规范验收

当前范围：[宏专题](../../../docs/implementation/tasks/compute/compute-graph-macro-normalization.md)。
产品候选 `026d40bcfdde389c9f69b2458b55d8157774ebda`，baseline
`a3e6046b053c7a40dc52c703ea049a38ec7795d5`；其后只改测试、文档和状态。
Windows是产品真相；WSL／Mac只前向同步。CUDA本体不要求Mac。

## 结果

- 12个产品文件的OS／compiler ABI／ARM条件统一至已有Core facts，显式include。
  私有CPU AVX2 program使用既有HYLE_EXPORT，补owning include目录；Native补直接PRIVATE core边，
  Core本来已在闭包中。API、layout、原数学、限值、原软件资格及43 CPU规则不改。
- [完整分类账](complete-ledger.json)：432条预处理条件、440条CMake条件、11个feature定义。
  产品原生重复facts与整数fact presence误用为0；presence-only feature／static／build宏、原SDK／
  GPU／版本条件保留。关闭的feature继续未定义，不把0错当作开启。
- [十个正文等价证据](body-equivalence.json)：所有改动函数正文与baseline一致；export拼写另经
  真实Windows DLL和Linux SO的loader／ABI执行通过。
- Windows MSVC与clang-cl20.1 **x64**、Linux GCC14 x64、Mac AppleClang21 ARM64，
  Debug／Release的真实平台、编译器／ABI、ARM与公开路线枚举等价编译通过。
- Windows：12模块CTests、7consumer CTests、2ggml CUDA CTests通过，native声明277＋223＋32项；
  一个Graph过滤器匹配0，保留原结果并补未过滤Graph两门，不将0匹配算PASS。
- Linux：11模块门、CPU Direct／Graph、未过滤Graph通过。Mac：11模块门、CPU Direct／Graph；
  原F16错误与六个fixture skip按下述test-only delta补齐，未过滤Graph通过。
- Windows／Linux／Mac各source、搬移static、搬移shared，普通消费者均执行Direct／Graph **48项**
  通过；静态宏及GPU-OFF负空间正确，实际DLL邻exe／ldd／otool loader路径通过。
- Vulkan两个owning adapter实际编译，五门通过（含原source及MASM composition），native **25＋7项**，
  无skip；只签宏／编译／现有资源合同，不解除旧Vulkan性能blocker或签新增数学能力。
- 非测试Compute＋Graph **60,561行／253文件**，统一脚本自检PASS；第三方、tests和bench不计。

## 原失败与纠正

[原始归档](raw-receipts.zip)保留原配置／编译／测试失败、skip、纠正、平台、交付、loader和XML。

1. Native的显式Core include发现缺直接依赖声明；补PRIVATE core，没有新增闭包或runtime成本。
2. Mac系统Python3.9不支持hashlib.file_digest；只改临时同步器的archive校验实现，未改产品。
3. GCC报旧GTest条件无花括号dangling-else；`7c85a977`仅补测试花括号，数学／规则不变。
4. Mac旧F16投影测试遗漏成熟ATen、对double求和的近零输出作相对比较，并保留已撤回的ARM F16
   累加例外。原／新ARM事实和正文完全一致；ATen两输入先转F32，原MatMul未改。
   `d1ea2845`仅迁移consumer测试到现F32合同：ATen合法身份、已有expanded-GGUF输入规模
   roundoff界（16×F32 epsilon×max activation×row L1）加storage ULP；删除半精度例外。
   不改全局half helper、不扩大产品支持、不补自研数学或重跑P1。Mac具名F16与六fixture实际补跑通过。
5. 初版CPU-only SDK consumer误要求未选GPU头；按既有安装裁剪合同检测header availability，
   不改安装政策。源码能单独编译OFF头，CPU-only SDK不伪造GPU API交付。

## 七门与限制

| Gate | 本专题结论 |
|---|---|
| scope | M1–M8、K1–K3全部处理；不恢复其它路线施工队列 |
| architecture | Core已有事实唯一来源，实际owner／直接边明确；原接口／资源与失败／生命周期保留 |
| correctness | 四compiler frontend事实等价、三平台模块与代表consumer、原失败／skip缺口实际补齐 |
| delivery | 三平台三交付、搬移／offline／loader／static宏／GPU-OFF负空间通过；Windows CUDA与Vulkan owning ON编译通过 |
| usability_docs | 普通用户无新增宏或内部target配置；恒定义facts用值、feature按presence合同，CMake目标变量分层 |
| red_team | 0匹配不算PASS、skip具名补跑、CPU数学不因旧fixture改写、原失败保留、镜像不反向覆盖Windows |
| engineering_performance | P0正文／事实／成本等价，无新runtime字段、分配、查询、搜索或计时；不跑P1、不新签性能赢家 |

本记录关闭宏规范有限范围，继承既有CPU数学／性能／清退结论，不代签CUDA剩余接线／Linux carrier／
三交付／完整模型region／best-of／自研清退，也不解除Graph／Vulkan的旧BLOCKED。
原U3 G1/G2 owner继续原有限任务；IREE既有能力仍是CANDIDATE，不因宏专题变成完整路线CLOSED。
reopen：新可复现目标facts／ABI／feature传播／export／include缺陷；不因未知硬件缺统计证书重开。
[机器摘要](receipt.json)及[selected plan](selected-plan.json)给出真实范围；selector READY不代替全部门PASS。
