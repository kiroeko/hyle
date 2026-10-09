# Vulkan V2：原 MLX packed 十六需求

<!-- MODULE REVIEW: SCOPED -->

baseline `a2c0d0c84a7e25a72617cf9a8641ccb75fac1f2b`。父轮
[Vulkan parity](vulkan-cpu-cuda-apple-parity-closure.md)；现有 owner
`compute_provider_iree_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`。
上批 [artifact 接线](vulkan-parity-v2-artifact-bindings.md) Windows 微型链资格继承，
不能当作父轮完整 catalogue/三交付/P1；本卡尚不授权新增产品改动。

## 当前真实需求与原源码审计

[共同端口](cpu-final-demand-carriers.json)含8 MX8、2 MX4、6 affine4，共16个 packed
合同；包括 selected experts、129280×4096 embedding 与 output projection。
自然 C++ 仍是 `mlx_quantized_matmul`、`mlx_grouped_matmul`、
`mlx_quantized_embedding` 的 Vulkan prepare/Graph compile，不增加 loader 或 custom callback。

原 MLX `2b2bb49790158ecd008d1685c94e3ef119792a00`，MIT；已登记
`third_party/mlx_cpu_quant/SOURCE.json` 的原 `backend/cpu/quantized.cpp`、`ops.cpp`、
QuantizedEmbedding；CPU/CUDA借用的相同 tensor 序列已在当前四路线目录中。
本轮只读这些明确源码，不重审其 CLOSED owner 或重新调查模型。

- affine：原低 nibble first U32 packing；code×scale+bias，F32矩阵累计和最终声明 dtype。
  mixed A/scale dtype 按现有 schema 提升 F32，准备时匹配，必要转换显式计费。
- MX4/MX8：原 FP4 LUT／E4M3FN；逐 K group 先点积，再乘原 E8 scale，再顺序加到F32结果。
  原 `dequantize_scale<T,32>` 将 E8 按目标 dtype 转换后乘；code0 为2^-127、255为Inf。
  不拿 IREE f8E8M0FNU 的255 NaN代替原 MLX255语义。
- grouped：只展开当前 group 与 selected expert；不展开整库或完整 selected K bank。
- embedding：先 checked row selection，再原 affine dequantize；只输出所选行。

主纵切是 packed projection/专家与真实 full-vocabulary selected embedding。
正交纵切是混合F32/F16/BF16精度、索引错误/重复和 changed payload 的通用 prepared tensor。

## 拟定有限 FIX 与验收预算

M1：原标准 tensor/arith/linalg/scf 程序承载上述三格式/linear/grouped/embedding；原
IREE compiler/runtime pin不变，MatMul和内部调度交给原 compiler。先验证动态group循环
和有限工作空间；不自写GLSL/kernel、量化器、GEMM或compiler pass。

M2：私有 ProgramKind/shape/原 scalar dtype 和原 native views；复用VM/context/queue/
fence/retirement/quota。grouped flatten `[T,S]` 只改native view的shape，不变payload。
最多五输入，继承上批六输入容量；不建立共享万能执行器或逐kernel vtable。

M3：普通 Direct/Graph hook 与冷feature/descriptor/32-bit indexing/workspace/grid判定；
mixed dtype复用已资格native Cast。checked IDs复用既有index guard与checked publication，
不能让非法id进入native gather或让失败partial结果覆盖caller目标。

M4：原序列/许可/离线制品与manifest/vendor digest；仍只在普通Vulkan配置交付，无运行时
compiler/Python或额外用户标记，DLL邻置不变。三交付和Air按父轮统一验证。

M5：成批普通两入口三格式/三dtype、mixed affine、group32/64/128、非方阵、多group、
two slots、完整A/weight/scales/IDs更新、重复/坏ID、旧lease、cancel/close/OOM、限域拒绝。
模型actual geometry必须另列真实GPU证明；微型source chain不签16全geometry完成。

P0记录one-group decode、原 dot/scale/累计、原VM内部循环和实际空间/复制/完成账单。
单完整carrier不制造性能竞赛；稳定父轮candidate再一次适用P1。原HYLE数学继续冻结。
实现/来源预算90分钟，资格/纠正45分钟；批末一个不可变候选build2/test1、重型串行。
具体新source/call/ABI错误只做一次具名纠正；未合格不能PASS，不无限compiler flag试错。

当前：SCOPED；原source与接口已识别，native动态group/typed workspace审计尚未完成。
