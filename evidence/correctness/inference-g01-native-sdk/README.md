# G01 Vulkan 驻留读取的 installed SDK 交付

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，
[receipt](receipt.json)和[raw.zip](raw.zip)记录实际输入、commands及所有原失败。
源码资格见[原生精度](../inference-g01-native-precision/README.md)。

static/shared均安装到fresh staging，再真实改名relocated，原staging不存在。消费者是
candidate公开Graph Vulkan测试与必要oracle helpers的复制文件，find_package只给安装
prefix，产品头/库从installed SDK取得，不使用HYLE私有类型或源树include。
F32/F16 D64/128/256、B2/GQA8/2、effective length/offset/padding、source只读、两slots、
旧ResultLease/Tensor、释放后重放均通过原oracle/容差；同源码旅程覆盖两种交付。

首次shared安装漏建所选产品导出的compute_state，补建原target后同staging续装，原FAIL
保留。测试oracle的half bit-carriers不属于Vulkan-only产品：只复制candidate types.h到
测试自己的oracle_storage.h，改helper include；产品不添加SIMD依赖、oracle计算不变。
首次缺header的consumer build FAIL也保留，consumer实际输入hash明确记录差异。

生产者复用纯Vulkan树与原锁/依赖；static新增patch source identity按实编译，4 jobs。
这不签CPU/Apple/CUDA SDK格、sanitizer、P1、全部G01或goal结束；其余验收继续活动卡。
