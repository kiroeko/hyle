# G01 native ragged 原直接消费者回归

候选280，实际Air Mac16,12/M4、AppleClang21；原selected CPU+Apple/CPU+Vulkan
source-parent工程及source-build复用，只构建g01_ragged_direct_consumer，四jobs。
[收据](receipt.json)和[原始命令/日志](raw.zip)保全部stage/失败/累计耗时。
原route_symmetry_{apple,vulkan}.cc覆盖有限shape/control/状态、独立slots/旧output与
CPU/native真数学，宏只启用已存在hybrid测试，未扩大普通全局算子或运行矩阵。
本门不签新的native分页、全模型吞吐或未运行的Windows GPU资格。
