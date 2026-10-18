# G03 CopyBatchUpdate 六组宿主 sanitizer

2026-10-14，按[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
固定candidate bcaec010b；[累计](receipt.json)、[实际末段日志](final.log)、[输入字节](inputs.json)、
[所有编排/编译flags/原始日志/明确模型](raw.zip)。Air Mac16,12 arm64 macOS27.0.1 AppleClang21；
原foundation-f5-native-host路线/模式树，最多4jobs，六组heavy串行。

Apple/Vulkan各ASan、UBSan、TSan通过：各公共初始化+64dirty updates，CPU原5个copy
source测试+新CopyUpdate published/坏ID/unaccepted/inline/原ordered guard测试6项，
Apple五生命周期+三个实际original native copy后继、Vulkan六生命周期/unknown原fence。
实查compute transfer/leases、对应native copy program/recording/run及每个fixture编译命令
具有相应-fsanitize，未用不加仪器的consumer冒充。没有重复已闭attention数学/P1。

ASan detect_leaks=0，不签LSan；TSan沿已独立证明的原final atomic acquire与Apple原
MTLCommandBuffer commit/completion显式验证模型，模型原字节封存，无suppression/新模型。
不签shader/driver/私有pool内部、CUDA runtime或完全原生GPU执行的TSan资格。
原脚本Python3.11 f-string语法FAIL0.014s保留；修复编排后六组最终PASS，产品未改变。
打包首次漏hyle路径前缀失败，不影响已完成测试；按实际文件库存重新封存模型。

累计170.768秒，包含语法FAIL，不重置原预算。
