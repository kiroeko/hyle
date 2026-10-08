# Apple DSpark：有界数值责任定界

<!-- MODULE REVIEW: AUDITED -->

父轮[Apple parity](apple-cpu-cuda-parity-closure.md)，baseline `7d36958ed`。
只涉及既有Apple测试/oracle，不重开CPU/CUDA或其它owner，产品FIX零。
两入口Pro0813/Flash0731已有相同残余；FMA切换没有解决，不继续盲改参考。

## 固定一次捕获

只用Direct两个失败ID、原六replay和actual-state反馈，一次编译、一次运行。
保留原逐元素/精确/完整输出门，诊断不返回替代PASS，不改输入、源权重、scale或容差。
捕获已经发布的attention、inverse rotation、grouped output与final output；只在测试回调中
分析，GPU控制/产品路径不变。输出逐层统计和首个差异位置，不保存巨型重复模型数组。

1. 由固定位置/频率计算原MLX GPU sin/cos，比较CPU libm和GPU系数；以两种F32收缩方式
   验证actual attention后的同一旋转方程，仅用作原因诊断，不选择更近参考签资格。
2. 独立复建原稀疏BF16 wo_a两个source terms，计算actual inverse输入的grouped equation；
   判断未匹配是否由前驱误差传播或该成熟乘法自身引入。
3. 固定原FP8 codes/scales与actual grouped input，分别比较原FMA/unfused F32参考；
   本次只裁决机制，不改变whole-fragment oracle或逐元素门。

最多一个具名捕获接线纠正。结果后停止重复捕获；由证据决定是否产品接线缺陷、原native
舍入合同或测试比较口径问题。若仍不能完整准入，具名保留未决，不用剩余通过项代签。
既有矩阵/三交付不重跑，不跑P1，不调GPU/系统设置，不扩性能或公共治理工具。
