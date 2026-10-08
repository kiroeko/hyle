# CUDA U3：Windows CPU+CUDA shared 交付资格

普通CUDA配置仅覆盖`HYLE_COMPUTE_CPU=ON`后，source产品、install／搬移SDK、offline
独立configure／link及两条独立路线的实际运行均通过。GPU Direct／Graph交付检查166／151项
与CUDA-only shared的同一声明范围一致；另有CPU FP32 Direct、原Q8编码及CPU Graph重复执行
三项通过。两路线各自执行，未实现模型级CPU+GPU分担。

GPU检查保留60个真实角色／编码配对、全34 packed、完整族／current／三sidecar、本地Qwen
与普通寿命边界。全部形状数学资格按不可变源码／算法与native／static证据复用，不冒称
本格重新跑了全部559形状。SDK所需DLL邻exe，运行PATH仅系统目录。

CPU consumer首次FP32用例未提供oneDNN已声明的caller workspace而失败；夹具按
`workspace_requirement()`分配后通过，原失败记录保留，没有改CPU算法、路由或旧资格。
source consumer与SDK实际编译、运行，不用设备枚举或copy替代数学链路。

精确身份、raw日志／XML／哈希及夹具源码见[receipt.json](receipt.json)与`delivery.zip`。
没有P1／性能WIN。混合static、Linux shipping和U4／U5／U6仍待；本子格不签整个U2／U3关闭。
