# H2 Apple 接线语法批次

这只是原生语法/公开头证据，不是数学、link、资源寿命、重叠或三交付资格。

- Windows来源：40db759573fb9c551d3deec6ab46f40078a91cd6；修复6c415ae968827a04985f5aac94161fa0e1fbf847。
- 实机：MacBook Air Mac16,12/M4/16GiB，macOS27.0.1/26A434，AppleClang21.0.0。
- 固定7个changed/product/consumer TU和4个self-contained header TU，串行clang++ fsyntax-only。
- 初始9/11通过；apple_memory.mm误用HostMapping.data，apple_weights.mm的Status/ResultError条件表达式歧义。
- 两处源码修复后仅补查两个失败TU，2/2通过。其余9个TU/全部公共头内容未变化，复用原批次。
- 原native owner/type/batch文件和日志保留；没有重编vendor、改公共仪器、重跑H1或P1。
- Mac可用页面每单元前约6.6GiB，4GiB停止门；无重型验证或数学运行结论。

[初始逐TU命令](syntax-result.json)、[有限修复结果](syntax-fix-result.json)、
[实机身份](machine-before-syntax.log)以及对应raw log记录实际退出码。父Windows取log时出现路径
拼接错误，但Mac整批已terminal；先读具体result/ps再取回同一批，没有因wrapper失败重跑原批次。
