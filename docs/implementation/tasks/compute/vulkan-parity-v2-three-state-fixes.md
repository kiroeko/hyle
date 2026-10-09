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

S1/S3已批接并编译：RotaryReference显式传过attention/compressor/独立方程，
旧频率、F32输出与原gate保持；原CPU/CUDA默认分支相同且无跨TU定义差异。
Flash两个实际制品独立参数行；完整prefill/三stage/全词表步骤未删除。
下一集中run仅每入口GGUF90/103与Flash Q8/BF16四行，原通过域不重跑。

`da1d8382` 每入口四行3 PASS：GGUF90/103与Flash BF16完整资格，Q8仍FAIL；
原BF16此前未运行现在独立通过。S2一次实际prefill RMS边界捕获：四层与原
ggml差0，独立F32参考最大4.77e-7至7.15e-7，已观测下一Q8_1编码两层各
1 byte变化；源GPU归约顺序与原scalar参考不同。仅Q8 Model增加显式native
NormReference参数，冷构造原图计算参考，CPU/CUDA默认scalar与数值门不变。
只剩Q8两入口一次定向完整三stage/词表/feedback验证，原通过域不重测。

### S2-N 结果与 S2-R 具名参考缺陷

`04ba306b` Q8 两入口完整 delta 均 FAIL；normalized 最大误差0.0529–0.0587，
原门0.04，无 validation error。prefill RMS 源顺序差已证，但它不能唯一解释完整
三stage 的误差，不把机制证据改写成完整 PASS，不重复该候选。

随后 source 审计发现独立的具名参考错误：Flash sidecar 的 `rope()` 输入、输出为
F32，整个模型使用 F32 graph state；Vulkan 测试共享 `HYLE_C4_ROTARY_REFERENCE`
却硬编码 `native_rotary(...bfloat=true)`，错误套用了官方 BF16 主干的存储边界。
S2-R 只通过 Model 的显式 RotaryReference 绑定原 F32 native 旋转，默认 CPU/CUDA
方程与产品不变，不改变误差门或完整片段。该错误的一次纠正预算15分钟；因两
sidecar 参考都受影响，只集中每入口 Q8/BF16 四行，不重测80/560/25全集。
失败则冻结准确未资格范围，继续父轮其它独立验收，禁止追加同原因试探。

S2-R `f2ef3389` 两入口 BF16 PASS；Q8 的完整三stage region `0x0c040122`
及 prefill/embedding 均通过，唯一错误转移至全词表 head `0x0c040125`：
0.0414580703 > 0.04。已定位第二个独立 fixture 缺陷 S2-H：head 实际输入
`observed[step]`，oracle 却取 `program.normalized_reference[step]`；两者是完整
程序已验收误差范围内的不同输入，后续 Q8 codec 放大差异。改为以实际输入
构造独立原 projection reference，全部三stage/feedback/reset门、全129280词表
行与head数值门不变。这是同输入正确性修复，不是放宽最终门或删除计算步骤。
CPU/CUDA 产品不改；共享测试 fixture 统一修正该输入不一致。仅两 sidecar
每入口一次 delta，15分钟纠正预算内结束，未过不再追加同原因试探。
