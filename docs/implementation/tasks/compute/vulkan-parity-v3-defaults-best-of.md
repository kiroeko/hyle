# Vulkan V3：成熟默认、条件绑定与有限成本裁决

<!-- MODULE REVIEW: SCOPED -->

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
