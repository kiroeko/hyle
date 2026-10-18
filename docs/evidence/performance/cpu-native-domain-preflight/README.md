# CPU Native 批量性能前置：部分资格，整体失败保留

产品数学固定为 `775b2bedace019de423edc5d8a6477772b8ef3ec` 的已验收 static SDK。
普通成熟 peer 实际分布为 oneDNN 24、ggml 146、ATen 108；278 个数值预检全部通过。
这不是完整模型推理或全域数值证明，既有关闭卡的语义、资源与交付资格仍是基础。

[原始收据](receipt.json)和[原始日志](preflight.zip)保留本次所有实际尝试。
首次短对照失格后，仅作一次测量源纠正：两臂采用同一个 SessionRunner 类型，并统一为
16 epochs、最低 2ms。一次错误并行启动导致重链接失败且跑到旧 exe；该尝试无效，原日志保留。
最终成功重编译后的 8 个完整 A/A 进程覆盖 Native/成熟路线、两种执行和两种分配顺序，
caller CPU2、单线程、above-normal，真实 submit/wait。没有更改用户电源或服务。

## 裁决

- **整体前置 FAIL**：72 个场景未满足原定单轮 10% / 分层中位 5% A/A 范围。
  不将它们签为 WIN、LOSS、PARITY 或性能 PASS；不增加采样、改阈值或优化自研。
- **206 个场景满足同一固定对照条件**。正式 A/B 前固定仅这些场景的合同，失格清单完整保留。
  这是按测量资格分批，不依据未采集的 A/B 胜负选择场景；正式 P1 仍需统一 controls 和独立采样。
- 72 项在去留审计中按既定“无合格完整优势则优先成熟”的政策处理，而不是假定 Native 性能落后。
  独有的严格资源或整域能力须另有源码与消费证据，不能借这项理由 blanket 保留整个目录。

资格复用绑定 frozen benchmark source、SDK 输入/运行 DLL、同源码 peer、完整原始记录 hash 和
最终 loaded `.text` hash。fresh consumer 重链接若仅 debug/timestamp 改变而 loaded code 相同，
复用此前的 A/A；code 或 SDK 改变则拒绝，不补跑寻找 PASS。统一 runner/stat 核心未改。
四执行/分配 strata 是本安装 artifact 的范围；没有第二个代码布局，不宣称布局无关或跨平台 SOTA。

rows32 是几何 holdout；统一 runner 的 sample-partition holdout flag 仍为 false。
普通 alias、严格内存、whole-region 与完整模型不因这个有限 leaf replay 自动获得性能结论。

前置之后的一次正式 P1 已完成，见[统一结果](../cpu-native-domain-verdict/README.md)；本页的整体
前置FAIL和72项失格不回写。产品删除/路由有限 FIX 尚未据结果固定。Compute+Graph 实现仍为
61,386 行 / 249 文件，不含测试或第三方源码，本目录是验证资产，不能计为产品完成。
