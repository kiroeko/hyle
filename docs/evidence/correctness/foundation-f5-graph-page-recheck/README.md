# F5 Graph页权限冷复杂度修正复核

[首批合同与结果](../foundation-f5-graph-page-first/README.md)不改写。本次冷alias拒绝只需
检查每个physical instance是否直接指向projected根：任何间接链必然包含这样的direct
child，因此仍拒绝全部descendants。避免逐instance再遍历完整链，alias检查O(V)，只在
显式bind冷路径；原ordinary submit无变化。

修正后的actual WSL CUDA Graph原12oracle、新F16/BF16页append/F4旧新旧六oracle及
三种真实negative Graph全部再PASS；Win原source/static链接运行、modified Graph TU/WX
通过。两个独立Graph头API未再改变，复用同根已通过的头证据；common32项/八compute头
实现也未改变。准确输入hash、原失败、所有累计build/consumer/header预算及原命令收在
本receipt/raw archives，未重置预算或重跑未影响算子。仍AUDITED，不签SDK/san/完整F5。
