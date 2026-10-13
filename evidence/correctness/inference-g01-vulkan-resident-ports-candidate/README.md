# G01 Vulkan 直接驻留端口：候选交付与宿主风险

不可变candidate `3d92eee97a70f7788dfb9bbcda73f6e0f801959f`，native runtime/a400不变。[收据](receipt.json) 与
[raw.zip](raw.zip) 保留所有实际阶段/日志、原compdb和2013-file单向canonical输入
manifest。两个变更TU和其余Provider/Route/runtime输入均对Git candidate SHA核对。

实际Air的source9项公开测试、搬迁static/shared SDK9项公开测试均通过；宿主ASan、
UBSan、TSan各通过9项公开+4项native风险测试。六个实际编译TU的插桩分别核对，包括
新增port/origin两处产品改动及原native runtime。source copy3反证FAIL、编译FAIL和
最终copy0见[stage](../inference-g01-vulkan-resident-ports-stage/README.md)，没有删除。

SDK普通find_package(Hyle)+公开Graph/Compute，无HYLE private消费者接口。主/留出
数学、readonly、slots/旧输出、64步origin/owner重绑、非vec4显式Copy和提前错origin
拒绝均使用原oracle。Windows签相关增量编译，未伪造RR/VP Windows GPU风险重签。
host sanitizer不签shader/driver，ASan不签LSan。P1与整个G01仍待，不由copy0或测试数
推导吞吐完成；两次原合格P1 FAIL都保留。
