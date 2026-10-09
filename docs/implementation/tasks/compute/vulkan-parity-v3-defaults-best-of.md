# Vulkan V3：成熟默认、条件绑定与有限成本裁决

<!-- MODULE REVIEW: AUDITED -->

baseline `755dec4c5`；owners `compute_route_vulkan`、`graph_route_vulkan`，必要的
原region接线归 `compute_provider_ggml_vulkan`。V2 Windows 104/559/80/34/45/八族及
原IQ1 bank资格继承；CPU/CUDA/Apple CLOSED、共享Compute/Graph与公共仪器不改。
主纵切仍为当前固定模型片段，正交为公开typed语义/失败/未知条件默认。

## 审计输入与有限范围

- D1：`vulkan_internal.h::route_facts` 未被任何prepare/compile使用。复用共同
  `RouteSelector/RouteMatchInput` 接入现有冷承载，不让用户排序或提供timing；
  平台、实际device/driver/API、编译器/源码版本、provider版本、模型/制品/phase/region
  和每个端口保留条件表达。未知条件匹配合法成熟默认，真实capability仍由原承载检查。
- D2：清点native tensor中的数学与存储/检查机制，按原ggml/IREE source和完整合同
  判定重叠/唯一/冗余；成熟完整替代无明确优势的self退出，不为保self追加优化。
  有效存储、NaN/RNE、checked索引、资源、未知device独有能力不误删，SIMD不动。
- D3：只对真实同合同重叠或Graph消费者成本运行有限P1。旧Recorder直接路径由caller
  供输出、Graph返回owned lease，不能直接比较。先冻结完成/输出拥有/失败/临时成本
  等价的原生peer旅程；没有同合同就如实NATURAL_ROUTE，不能为通过临时删掉服务。
- D4：API/算法/default稳定的不可变candidate才一次P1，先相同workload A/A与双顺序；
  事前paired口径、n15、5%门、placement，源/成本先行，失败不追加样本或改系统模式。

source/接口审计预算30分钟；有限批接30分钟，纠正20分钟一次。P1最多一次正式批次，
预检不合格即停止该比较，单carrier不造竞赛。适用未验不能PASS，AA失败不认产品FAIL。
Windows同producer build2/test1、重型串行、DLL邻置/隐藏child；已过V2大矩阵不重测。
验证只新默认/条件/真实重叠、必要资源与直接Graph消费者delta；三平台/三交付仍V4。
必须selector、closure明确previous-ref、docs/diff与统一LOC。本文SCOPED前段只读，
具体原算子漏绑定及删除表固定后再AUDITED，不扩模型、算子研究或公共治理。

## D1 有限FIX已审计

现有RouteFacts保留vendor/device/name/driver/API/subgroup，但未接线、缺编译SDK/原
provider源码版本，空文本还会成为InvalidFacts。仅修改route owner：Environment冷事实
补SDK/ggml/IREE源码身份，route_facts改为稀疏有效事实；既有所有原typed/capacity
admission前接入固定ggml/IREE默认selector，PlanStep输入/输出、workload全字段传入共同
matcher。默认selector不约束特定机器/模型；窄条件可用共同matcher，未知facts不得命中
窄规则、仍可匹配合法成熟默认。GPU代数未可靠提供则未知，不猜SM或借CUDA探测。
matcher只prepare执行，native执行状态/资源/融合形状不变，没有新用户偏好或hot metadata。
Graph实现说明的旧十格式名单同步当前原承载范围，不改接口责任。

source/contract审计还证明Relu/Gelu原shader虽存在，但完整公开NaN/Inf/负尾语义不同；
不能凭正常有限输入计时认完整替代。native低feature/严格存储等独有域暂不删除，D2表逐
域继续源裁决，只有真实完整重复才比较或清退。D3的旧不等责配对仍不运行。

首批验证只新条件适配的组合字段/unknown facts/default、两入口ggml/IREE实际链与
workload保留、Graph报告和必要重放；V2相同typed/math大矩阵不重跑。公共matcher与
共享模块不改，CMake只配置现有生成头里的锁定事实，不新增依赖/用户开关。
