# H3 shared 安装包

产品 `97f9332ac`，source/shared 分别有独立 receipt。Windows 与 MacBook Air 由原 source shared
对象生成安装包，staging 真正 rename 到 h3-shared-sdk-relocated，原 staging 不存在。
使用 `find_package(Hyle COMPONENTS graph)` 和 cpu;vulkan 集合，五个独立安装消费者通过：
普通 Compute、普通 Graph、Direct migrated-weight MatMul、组合 Q8 FFN/正交管线、资源与寿命 contracts。
三个新增公开头在安装 target 的真实导出 macro/include 下单独编译通过。

Windows app 的 DLL 由 hyle_stage_runtime 放在 exe 目录，不依赖工作目录或额外 PATH。
两端没有重跑已经通过的 source GPU overlap；此处未签 static 或 WSL GPU运行。
实际构建/安装/消费者 argv、退出码、资源 guard 与路径在 receipts；日志保留完整测试结果。
