# F5独立component断言修复与原合同验收

完整公共component合同实际PASS，总elapsed 856.297秒≤当前1200秒；原上限300秒，
用户2026-10-13明确授权同root追加300秒且类似验证不再逐次询问。原失败/累计不重置。
首次缺Graph batch test ownership、第二次graph-only误要求完整Compute façade、第三次
private runner漏Vulkan SDK环境的FAIL与原时间完整保留。没有新root/重取预算或回放旧算子。

模块测试归属已由真实三项共同Graph测试+batch.h独立TU修复。独立工具修复只把
graph-only parent的required façade清单从compute/graph改为graph，全部CPU实际factor、
native缺席、其他组件/安装/依赖断言保持原脚本；不改产品DAG、closure checker或selector。
原源码声明只在全部root被选择时产生完整façade，tests依赖不是tests-off产品授权。

脚本虽首行写Configure-only，实际也build/run原Core消费者、build container_intrusive/
io_transfer测试；本次按完整原脚本执行到底，不把仅前缀通过称整个PASS，不改公共工具。
命令、全部terminal/原失败/hash、实际脚本、唯一断言差分、SDK环境与预算见[receipt](receipt.json)/raw.zip。
固定F5数学25e与IREE san配置11c348e40；此PASS只完整component合同，不代签SDK/native san/P0/最终closure。
