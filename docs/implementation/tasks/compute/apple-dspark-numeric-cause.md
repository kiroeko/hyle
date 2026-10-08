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

## 捕获裁决与唯一后续 FIX

`4ac59a7e`一次Direct两ID/六replay完成，原门仍FAIL。原MLX GPU与CPU libm的224组
sin/cos有162组系数差异；Flash原5个exact inverse差异在固定GPU系数的同一方程下全部
消失，两种收缩方式都逐值一致。Pro/Flash全部12次grouped输出在actual inverse输入下与
独立原两term BF16方程逐值一致；全部12次final FP8在actual grouped输入下与原独立
FMA/unfused参考均逐值一致。不存在这两层接线错误的证据；Pro全链4项残余是合法前驱
差异进入消减/后继量化后的比较，不能把局部PASS代签whole PASS。

唯一有限FIX是测试参考完整绑定原MLX数学：旋转系数由独立原MLX GPU计算并在构造fixture
前固定；SinkAttention由独立原MLX公开fast API承载，使用独立reference生成的query/KV和
独立cache状态。所有模型输入、源权重、codec、CPU独立FP8/grouped方程与各BF16存储边界
保留；不消费产品lower/Graph lowering或candidate中间输出构造完整期望。仅原局部inverse
checker继续用实际前驱，其设计已在原C4明确。CPU/CUDA默认参考不变，Apple不逐值择近。

primitive独立数学门与source/storage证据仍保留；这次补的是同原native精度的完整接线参考，
不是改模型数学或放宽原whole-fragment逐元素/状态/reset门。两个完整witness唯一一次正式
复验，两入口串行。若仍不通过，停止本方案并列具名未决，不继续换参考。产品FIX仍零。
捕获源码随证据归档并移出常态测试；公开头9/9已通过，目标hidden/confidence两入口各2/2
定向确认通过，既有矩阵和三交付不重复。
