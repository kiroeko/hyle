# G03 CopyBatchUpdate Windows 定向验证

2026-10-14，按[活动卡](../../../docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
固定candidate bcaec010b；[实际结果](final.log)、[累计记录](receipt.json)、
[产品字节](source-identity.json)、[原始日志与编排](raw.zip)。原cached
C:/h7/inference-g01-native-vulkan/build，MSVC /W4 /WX /utf-8，最多4jobs。

只构建g03_copy_update_public及其compute直接依赖，actual CPU初始化+64 dirty steps
通过，原generation/coldslot/metadata保留、foreign pair拒绝且不消费、同一writer只返回
一次，原staging readonly回读比较全16384bytes历史。公共/原Vulkan route候选compiled
并链接成功，Windows没有运行Vulkan GPU或全模型，也不代签Apple/CUDA资格。
原parent重新configure与相关头重编计入252.547秒，两个阶段均terminal0；没有重写
构建系统或启动旧graph全局矩阵。三SDK/hostsan/其余生命周期仍由source卡继续。
