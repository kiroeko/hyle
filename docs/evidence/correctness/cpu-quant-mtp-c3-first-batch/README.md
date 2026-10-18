# C3 第一组 Windows 接线验证

精确候选/范围/剩余项见[summary](summary.json)，原始命令见[receipt](receipt.json)。
本目录不签C3整包、Linux、三交付、完整模型分支或整个goal完成。

- 11个新增GGUF格式在普通CPU Direct/Graph通过MatMul、GroupedMatMul和GetRows实际计算。
  bytes来自原block结构；对照是原decoder/CPU dot，包括原activation编码，避免不同量化语义。
- Changed replay、重复expert/embedding索引、越界写保护及失败后新slot验证通过。
  接受后失败按公开合同退役invocation；没有为测试改变产品失败语义。
- NF4/nested NF4、AWQ、CT int4原CPU解码序列的F32原格式差分通过；默认BF16/F16精度、cold constant
  完整消费、Graph和故障资源格仍待验，不能由F32测试外推全制品完成。
- 本批编译j2，测试j1，DLL邻置；复用唯一producer，没有P1或自研数学优化。

此前测试夹具纠正：自然views API、Graph add_node/result API、失败后新slot；标量/AVX2原Q2_K
重排的F32舍入以operand norm定界，GetRows仍逐值精确核对。不以降低模型精度或更改kernel找PASS。

MLX三种混合模式与压缩专家执行、C4新分支、全部准入格及两系统三交付尚未完成。
