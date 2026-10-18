# CUDA U2/U3：Linux 源码与搬移静态 SDK

原已完成批次现已封存；没有重新构建或重跑。source configure/build、profile admission 2项、
本地真实Qwen Graph 5项、reference生命周期回归1项、install、prefix搬移、offline SDK
configure/link全部通过。独立SDK的50组、Direct166项/Graph151项通过，无失败或skip。
命令、原日志/XML及哈希见[receipt](receipt.json)、[归档](delivery.zip)与[用例并集](sdk-test-union.json)。

使用GCC14/C++23、CUDA host GCC13、Toolkit13.3、SM89real；HYLE只从搬移后的SDK取头/库。
原ggml参考程序是独立fixture资产；Torch与CUDA叶子按已声明机器依赖cache取得。
运行不设置LD_LIBRARY_PATH，原ELF loader合同实际通过。原Torch CUDA12叶子与原native
CUDA13库共存的既有组合实际运行通过；不据此签其它组合/版本的资格。

60个实际GGUF角色/type组合、34个non-GGUF、全部必要current/family/state、本地原版及
普通生命周期覆盖ABI/交付接缝；完整shape数学复用同原源的[Linux native 651/649](../cuda-final-u3-f32-native/README.md)。
不能把60写成559或把shipping数目写成新的全shape资格。本批无P1、无新性能胜负。

Windows mixed static、Linux shared/mixed及U4/U5/U6仍未完，本页不签整个CUDA收口。
