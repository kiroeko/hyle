# Apple MLX/Metal：覆盖、接线、默认优选与清退收口

<!-- MODULE REVIEW: SCOPED -->

baseline `406cc3c23843ea9928e45970527ed2664f8f2728`，Windows工作树干净。
2026-10-08用户要求Apple按已完成CPU/CUDA相同原则开始。产品真相仅Windows；Apple副本可丢弃。
此页先是协调只读审计卡，不把旧AUDITED记录自动变成新scope产品授权。写产品前按实际owner
重新审计并独立提交有限FIX；旧预算、失败、CLOSED/BLOCKED不自动重置。

## 四个有限包

| 包 | 权重 | 完成条件 |
|---|---:|---|
| A1 当前目录/成熟承载/接线与构建审计 | 25% | 全104 GGUF/559合同、45plain/34packed、八族及官方更新/MTP/DSpark映射，真实来源、数值/资源/成本与有限FIX |
| A2 批量成熟接线与正确性 | 35% | 成熟库优先；按实际端口/编码一次批量接线后集中准入、完整输出/状态与资源验收，单承载不造竞赛 |
| A3 默认、有限best-of及自研去留 | 25% | 真实重叠且同精度/转换/同步/融合/完整终态的一次有预算比较；有条件规则及未知条件成熟默认，冗余自研物理清退 |
| A4 三交付与终审清理 | 15% | 当前Air source/static/shared/relocation/metallib/dylib、两普通入口、失配/关闭/失败、docs/closure/LOC/垃圾清理、推送后停止 |

## 需求与边界

需求仅来自[共同模型目录](../../../architecture/model-compute-catalogue.md)及
[逐制品端口](cpu-final-demand-carriers.json)，与CPU/CUDA对称，不重新研究模型/下载巨型权重。
包含DeepSeek Pro0813/Flash0731、官方混合FP4/FP8、主流已登记量化与必要MTP；不能以旧
Apple40/44或曾经的M4性能机证书代签完整新目录。HYLE不实现loader、整模型engine、CPU+GPU
模型切分、HIP或分布式；GPU的host控制/准备/同步和同GPU不同成熟程序组合仍在范围。

第一候选是原MLX GPU程序与ggml Metal codec/完整region，先查具体原版能力及接线成本；
不按Provider名字假定覆盖，不因漏接就新增HYLE算子。所有现有自研数学冻结，成熟接入缺口
排除前不扩张/优化/研究自研。确无成熟同合同承载时列具体缺口/取舍，不伪PASS。
MLX compilation/fusion、ggml原管理、MLX arrays/原生完成和Metal buffers保持，不引入
万能Queue/executor或逐kernel vtable。资源封装、语义边界修复和成熟组合不误删为自研。

## 当前真实事实

Air连接 `ssh kiro@172.22.10.198` 默认身份；2026-10-08实际为Mac16,12/Apple M4/16GiB，
macOS27.0.1 build26A434、AppleClang21.0.0、CMake4.3.4、SDK27.0，交流电，剩余约43GiB。
设备/OS/工具链与旧证书不完全相同，旧performance只保留历史，资格须重新匹配。
当前remote `~/hyle` HEAD为1bf0b556、工作树干净；产品源码不得从它反向复制。
当前route/graph owners为旧D5-F45 AUDITED，ggml Metal为旧CPU enum迁移AUDITED。
源码 `native_quantized_type`只接10种GGUF类型，default明确decline新增CPU格式；
`apple_mature_float`明确拒绝FP4小scale roundtrip。先静态逐合同映射，不把这些当新数值FAIL。

## 有界执行

A1：一批需求归并、一批原成熟源码检出、一批接线/构建审计，各具体问题最多一次follow-up；
首小时产出静态映射与有限FIX，不翻遍历史任务。A2按能力域批量接线/延迟验证，source低并行2、
测试1、重型串行；缓存复用，Mac磁盘不够先清拥有者精确临时路径，不删source/模型/共享cache。
只同步完整Git候选/archive并记录身份，禁止手选远端产品文件反覆盖Windows。
配置步骤只以CMake/参数/工具链为输入；普通代码变化不重新配置/重复全集。具名实现失败最多
一次定向纠正，再按协议裁决，不换目录/ID重置预算，不重写公共治理/仪器。

A3在实现/接口固定的不可变候选上用统一hyle-perf，5% material margin、固定n15，
Apple实际QoS/交流电/low-power/机器身份收费；不冒称P核hard pin，不升31。不合格AA项不进
正式性能，近似/噪声直接选成熟。只导入完整候选赢家，不把单kernel时间代替终态成本，
不拆成熟融合来竞速。不无限优化保自研；真正unique资源/语义具名保留，有完整成熟替代且
无完整优势的自研清退。SIMD本身不动。未测硬件/版本仍有合法成熟默认，不让用户排rank/timing。

## 预声明验证和停止

只读A1仅输入SHA/映射完备性、docs/closure/selector与统一LOC，不build/P1。
后续按有限FIX选择原算子/region、资源/lease/Completion/失败、两slot/变输入、index/encoding
边界及当前Air三交付；未变域按source/合同/工具链/机器匹配复用，不复制CPU/CUDA PASS。
Windows/Linux Apple-OFF独立公开头/构建合同按实际改动选择，不制造Apple binary journey。
每包完成/重大scope变化报告实现/验证/未完/阻塞与Compute+Graph全实现LOC，文档/状态不build/P1。
七门全部适用证据成立后关闭本轮并停止；其它路线/公共Graph旧Vulkan blocker保持。
推送采用[skip ci]，不使用Actions。2026-10-08用户明确选择建立独立持续goal，现已创建；
A1已实际产出[559合同静态映射](../../../../evidence/correctness/apple-parity-a1/gguf-static-binding.json)。
其中127合同/11类型缺ggmlMetal类型绑定，432仅静态有映射，全部runtime仍UNVERIFIED；
不把这一统计当Provider实际支持结论。下一有限动作检出现有成熟Metal codecs与完整操作的
适用精度/资源后固定批量FIX，所有自研数学冻结。
