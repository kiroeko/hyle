# C5 实际需求与待验收合同

固定 C1 的完整 header/index 生成本轮验收输入，没有刷新模型 revision 或下载大权重。
本目录是 AUDIT 输入，**没有执行 PASS**。

- [GGUF](gguf-demand.json)：104 档位、126,809 tensors，去重为589类合同。
  559类是投影、grouped expert/head、embedding/head、位置表或整数hash表；
  30类是非线性/卷积参数，必须回指完整模型族片段，不能制造MatMul测试。
- [非GGUF](non-gguf-demand.json)：四个已准入压缩制品、34类真实packing/geometry合同：
  nested NF4六类、AWQ七类、CT五类、MLX affine六类/mxfp8八类/mxfp4两类。
  NF4形状与本地官方BF16父权重header逐项对账；AWQ/CT/MLX直接校验packed/scale/zero/bias维度。
- [NF4实际quant-state](nf4-quant-state.json)：从固定仓库读取两个不同文本projection的小metadata，
  原header SHA与C1一致，实际BF16/group64/nested group256确认。只证明这两个witness，
  不假称逐张量metadata data section均已下载。

去重保留每个制品到tensor角色/shape/storage合同的关联。执行资格还须匹配typed端口、
activation/accumulation/state、实际Provider软件身份与两入口；相同dtype名称不自动复用。
官方新版与三sidecar的完整branch资格保留C4收据；旧模型族完整片段与新增角色合同必须对账。

计划允许缩小token/batch/驻留expert；投影K/N与非expert head group保持真实，
全词表head允许驻留分片，embedding保留词表跨度/末端索引。合成非零合法payload不是实权重模型推理。
剩余普通入口输出、失败/资源、身份及完整required region的执行收据尚待C5补齐。
