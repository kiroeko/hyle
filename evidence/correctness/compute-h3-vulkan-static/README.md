# H3 static 三交付与边界检查

库实现候选 97f9332ac（e9c5e2c3b实现相同），expert-ID补充测试d2d1e86e3；文档/状态提交不触发重建。
Windows及MacBook Air各自 static 安装包完成真搬迁，原 staging 不存在。五消费者全部通过：
原Compute/Graph、Direct迁移MatMul、组合Q8 FFN/正交管线、资源/寿命contracts。
三个新增公开头按实际安装target的macro/include单独编译通过。static没有制造新的GPU性能比赛。

Windows初次完整依赖构建1195.1秒，单线程owned peak 11.402GiB，12GiB保护未触发；消费者构建
21.3秒，运行3.0秒。Air原静态依赖构建单线程，peak RSS 2,834,235,392字节，消费者按实际receipt。
源/共享已通过的GPU时间线没有重跑。Windows DLL与所需原生资产经原hyle_stage_runtime同exe目录交付。

额外expert-ID测试只重编/复跑contracts：固定catalog从ID7改ID8，同一Graph重新bind/prepare后
输出从-2变-3，Windows和Air source/shared通过。静态消费者使用补充后的同一fixture。
新 ID 未改变Provider或重编Graph。

Win/Air还用真正的Vulkan-only和CPU+CUDA/Apple SDK验证缺CPU或缺Vulkan在find_package阶段拒绝，
没有删SDK文件伪造负空间。四个Vulkan公开头在路线禁用且无native SDK include下独立编译通过。
此处不签WSL真实GPU运行；WSL的编译/交付和已接受UNQUALIFIED限制另记。
