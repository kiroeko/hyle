# G03 CopyBatchUpdate 三交付公共 consumer 资格

2026-10-14，按[活动卡](../../../docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
candidate bcaec010b；[公共consumer](consumer.cc)、[实际结果](final.log)、[累计](receipt.json)、
[所有安装/relocation/独立头/编译命令/原始日志](raw.zip)。source四路线已另封存；本处
Apple/Vulkan各static/shared + CPU/CUDA static/shared，共六个SDK交付配置，8次公共65步。

MacBook Air Mac16,12 arm64 / AppleClang21、WSL GCC14.2 actual RTX4070 CUDA13.3。
全部仅find_package(Hyle COMPONENTS compute)+hyle::compute；未修改公共consumer字节，
仅宏选择真实原路线。producer来自原cachedSDK树，install至staging再移动relocated，
确认旧staging消失；编译命令只用relocated头、不引用canonical include。对应batch_transfer、
resource_leases、native memory/CUDA backend三头各独立严格编译，6x3=18 TU；installed
header byte等于canonical。每个run同一原copy programme/slot完成初始化+64 dirty updates，
全历史独立读回、版本/metadata/control回收正确，shared实际调用新增compiled traits。

没有Graph或私有hook作为public consumer依赖，也没有数学P1/全模型；native opaque
metadata/private driver成本未变，仍unknown。CPU/CUDA producer与此前source树分离，
HYLE_TESTING关闭，后续source-only event观察钩子不会进入SDK资格。本证据不签pending
attention/late native output，完整G03仍独立施工。

实际累计139.132秒（Air52.160+WSL86.973），全部terminal0。
