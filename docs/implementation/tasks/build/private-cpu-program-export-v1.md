# 私有CPU程序出口检查纠正

<!-- MODULE REVIEW: AUDITED -->

- owner：构建/shared export检查工具，独立于CPU模块产品；baseline `5fefc3d7`。
- 真实consumer：已搬移Linux CPU SDK的原完整AVX2程序，40个Direct/Graph格式/精度格与普通样例已运行通过。
- trigger：checker把`libhyle_ggml_cpu_avx2_ffa4e8b8_bdae1e43311ee486.so`中唯一的
  `hyle_ggml_cpu_program_v1`私有加载入口判成公共Native FFI；source可见只有私有loader调用，
  descriptor ABI仍私有，没有C public header/consumer/安装C产品。
- 有限FIX：只有包含当前commit/source身份的精确private program文件名与该唯一入口组合可免于公共C判据；
  仍扫描模块全部其它出口，不排除整库、不扩大hyle前缀白名单、不改产品ABI/loader/原库。
- 验收：既有PE/parser/公开C/allocator/fcontext/oneDNN拒绝与empty合同保持；
  private模块正确正例、同入口在public库、错误program名称/身份、private库其它hyle/allocator出口全部负例；
  实际Linux已安装shared扫描，不重建producer、不运行P1。
- 预算：一次小函数/测试修改、一次真实扫描；不重写selector/closure或性能统计核心。

本卡审计已固定该集合，不以工具PASS签CPU整体或三交付完成。当前F3见
[CPU承载卡](../compute/cpu-final-f3-platform-carriers.md)。
