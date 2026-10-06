# C5 非GGUF未压缩text合同

固定候选 `c740abc74a5a62ffacf60b26798228ca484317df`，Windows/Linux普通Direct/Graph各45项、
共180项通过，无失败和skip：[Windows](windows/receipt.json)、[Linux](linux/receipt.json)。
[需求](demand.json)从四个固定非GGUF制品的实际uncompressed tensor生成，保持dtype和内宽。

- 13 projection、10 norm、3 conv、2 head、3 embedding、3 position lookup、3 log-decay、3 time-bias，
  以及HC base/sink/router bias/I32 table，按实际kind/dtype/shape去重。
- Qwen zero-centered norm与普通gamma分开：原BF16/F16 weight→F32→1+weight→RMS→一次回原storage；
  gated RMS与DeepSeek普通gamma不加1。FFN/MTP/attention实际epsilon不使用API默认值代签。
- 全248,320词表embedding跨度/末端/重复index、全head行实际消费。projection真实K/N，
  head以4096行驻留分片；位置APE保持实际shape，I32 table不转浮点后计算。
- Native low-dtype卷积采用原ATen conv1d，先转F32累加、一次回F16/BF16；没有新增HYLE数学核。
  原ggml F32卷积不变。BF16 C6144/10240、F16 C10240、原[C,1,4] storage经View进入普通入口。
- indexed消费检查越界写保护、失败slot退役/新slot恢复。Native因子还执行真实原allocator OOM、
  独立并发卷积slot及metadata回收；Linux factor准确收据单独保留。

固定语言范围之外的963视觉参数没有扩进本goal；487 packing companion已由F2真实端口资格覆盖。
原全tensor header/index保存，排除不是删掉事实或降低本轮语言数学范围。

## 测试构造成本与资源诊断

原数据行有17或256的精确周期。逐元素ScalarHalf编码全词表使准备耗时明显；
只停具名自有C5 plain consumer，原未完成批次不记PASS。改为同字节周期行生成，
projection/Gather仍保留完整原payload、K/N、词表和实际原库执行，未更换数值门。
同候选普通源消费现在每入口约6–8秒，不把验证构造耗时当产品性能。

故障注入代理曾漏转发原CPU allocator的raw_deleter，小conv原raw allocation路径因此退出。
恢复原allocator接口后正式Native因子通过；此修正只在本地测试代理，不改公共性能仪器。
软件版本、资源、转换/临时与未知私有上界沿现有managed合同；不宣称capture、零热分配或严格RSS界。

两系统更新安装SDK与全范围聚合仍需完成；本目录不是整model RUN_PASS或goal CLOSED。
