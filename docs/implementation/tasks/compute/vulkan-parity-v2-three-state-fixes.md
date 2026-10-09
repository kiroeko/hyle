# Vulkan V2：三个剩余状态合同

<!-- MODULE REVIEW: AUDITED -->

baseline `2a7e840e13ed64fe6190afea7c44cad2f536853b`。继承[实际矩阵轮](vulkan-parity-v2-catalogue-matrix.md)
已资格的 80/560 两入口及当前状态21/24；不重测这些完整矩阵。
owners `compute_route_vulkan`、`graph_route_vulkan`、`compute_provider_ggml_vulkan`。
本有限批目前只改测试承载，产品仍冻结；CPU/CUDA/Apple 产品与默认参考不动。

| FIX | 具名当前问题 | 有限修改／裁决 |
|---|---|---|
| S1 | GGUF profile90/103 indexer/compression 的 Double RoPE 参考与原 F32 GPU phase/函数/产品不同 | 给测试 recipe 传显式 reference 参数，保留原 iterative 频率、F32 存储和完整门；禁止全局／TU 宏 |
| S2 | Flash mixed sidecar 的首偏差在完整三stage／norm／Q8边界，Q8失败使BF16尚未到 | 同输入首边界账单一次，原库原程序与独立方程对照；只修具体接线／原舍入定义，不放宽门 |
| S3 | 一项 test 内合并两制品使后项未验且难归因 | 两固定制品独立参数行、原完整程序/全部词表/反馈/reset保持；一候选集中必要delta |

S1 已有原source事实：旧 GGUF 使用 iterative 频率和 F32 输出；官方使用另一频率及
BF16边界。前一宏草稿因跨TU inline定义不一致已废弃，本轮只显式参数化测试函数。
S2 不预认是产品或参考错误；先定位同输入偏差。成熟算法/HYLE数学/GLSL均不扩张。

调查／批接45分钟，资格纠正20分钟；各具名错误一次纠正，通过则移交父轮剩余完整
branch/80参数/银行/资源与V3/V4，未过写精确FAIL/UNPROVEN，不无限重新计时或研究。
P0为原数据/精度/资源边界；无P1/性能宣称。Windows同producer build2/test1，GPU串行；
只选三个剩余合同及因新reference改变的必要直接消费者，原失败记录保留。

验证：GGUF两当前profile普通Direct/Graph完整attention/compression/MoE/hyper/head，
Flash两mixed制品prefill→共享embedding→完整三stage→全部词表分块、真实反馈/reset；
原数值／索引／生命周期合同保持。三平台/三交付/default/正式P1仍父轮未验。
公共API/算法/组件图/仪器不变；native reference拥有独立原设备/完成，参数冷期固定。
