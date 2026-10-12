# F5 原生错误、实际退役与D256定向复核

baseline23b1e6672，AUDITED有限GRAPH-FAIL-01、GRAPH-DOMAIN-01与原OBS/DRAIN批准范围内
UNKNOWN-REUSE-01。不是candidate或完整F5风险/三交付证书。Windows产品真相只向可丢弃
WSL验证树复制；原输入历史、FAIL/TIMEOUT、config/tree/root与累计全部保留。

实际WSL RTX4070/GCC14.2/NVCC13.3/shipping/static原native Graph门六episode通过：

- node1与node2各自H2D/plan四个真实accepted sequence后追加具名错误，原CUDA event仍
  负责退役；prepare传播错误，不submit混合token/metadata，不发布输出、不复用失败prepare。
- 原BatchProvider run真正接纳第一个region后给EnqueueResult追加错误；实际run计数为1，
  依赖的第二个region没有执行，原错误保留、不发布两个输出，drain实际guards/quota0。
- 两个region所在的完整queue segment已经接纳后插真实CUDA host Gate，原event覆盖Gate。
  unknown poll后释放公共Completion、close/drain；Gate前drain等待且input guards仍在，
  Gate后guards归零、原错误保留、不发布输出，quota0。80ms只判断阻塞，不作性能结果。

test-only same-build cold Provider/Backend table装饰只转发原callback并追加测试错误/观察，
configuration retain/release hooks保留；没有修改产品Provider、Graph runtime或增加测试开关。
原Graph把两个shared regions合并到一个queue sequence，不能用finish计数证明suffix。
首次fixture编译FAIL（namespace中的include、Status/ResultError ternary）与consumer-resume11
错误sequence假设造成的FAIL保留；其中第五episode的早期PASS没有真实run计数支持，不能
用于当前suffix裁决。最终收据以真实run计数与完整segment Gate为准。

同门额外D256 paged/ragged×F16/BF16四域实际经过HYLE BatchProgram/native slots与Graph
kernels，双node/shared metadata/two slots/update/old result对24个独立double oracle通过。
原D64 paged F16、D128 long ragged BF16 split/merge、页/F4与ragged多请求消费仍通过。
这是实际Graph承载资格，不把raw peer probe或此结果写成standalone Direct D256 SDK资格。

Direct新增第三unknown episode不先cancel/close，在原流同步且实际in_flight归零后重试prepare。
consumer-resume16干净FAIL：`prepare rejected=0`、`unknown failed native slot reused after actual retirement`。
原两episode先close再drain，不能证明这个不主动关闭场景。局部修复仅在prepare通过
pending/use/completion actual退休门后检查原native terminal并设置control failed；失败slot
不可重置为成功。没有新增字段/API、warm算法或callback对普通failed flag的并发写入。
最终Direct三episode全部通过：poll/wait+cancel/close/drain，以及open失败slot实际退休后
仍不可reuse；原首错与metadata quota0保留。

累计预算见[receipt.json](receipt.json)：Direct configure16.403/build481.533/consumer197.157，
仍是批准一次追加后的总上限360秒，旧TIMEOUT未改；Graph configure16.403/build91.935/
consumer15.994，原consumer180秒。重型跨机器串行、库按可用内存j4/j2/j1、consumer1。
没有转移原Direct失败到Graph门，新Graph门验证自己的完整接缝；UNKNOWN-REUSE仍原Direct门。

源码影响READY计划与原skipped reasons封存为[validation-plan.json](validation-plan.json)。
Win common33项、CPU Direct/Graph完整矩阵/页/ragged/118 OOM位置、实际RTX Vulkan Direct/
Graph原消费及WSL CUDA Graph定向复核通过。公头未变，复用匹配独立头证据；不回放全库/
原算子/P1或A/F4。Apple无本次API/product seam变化，Windows平台selector排除，不记新PASS。

仍未签：其他route适用producer/partial/native error/abandon/OOM组合、standalone Direct D256
installed消费、18 installed SDK格、适用san/P0/guide/分离红队与合法最终closure。goal保持active，
compute_batch/graph_batch保持AUDITED；不把本有限门改成全部六FIX完成定义。
