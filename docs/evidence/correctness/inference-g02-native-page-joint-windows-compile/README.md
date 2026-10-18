# G02 Windows CPU+Vulkan 定向公开链接

固定产品candidate `1cd0893979e1672a0f6fac98b783a6a3bfcabbca`；实际MSVC19.50.35728.0，
复用C:/h7/inference-g01-native-vulkan/build，验证parent首次启用cpu;vulkan，四jobs。
只编并链接g02_joint_public_compile：同一public joint consumer与三个standalone headers，
只链hyle::graph。首次CPU依赖含oneDNN/ggml，实际419.171s，build exit0；原candidate
产品/28输入SHA逐个核验，public+两个真实hybrid TU的实际compdb与exe SHA保留。

[receipt](receipt.json)、[编译命令](compiled-inputs.json)、[raw](raw.zip)保真实命令、工具链、
日志、测试输入与runner。三个头为graph/batch.h、cpu_vulkan.h、graph/cpu.h，/W4 /WX。
没有运行exe或GPU，不代签Windows数值/吞吐、原G02停止的Windows CUDA/Vulkan数学门。
