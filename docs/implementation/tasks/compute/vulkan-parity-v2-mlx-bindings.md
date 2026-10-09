# Vulkan V2：原 MLX packed 十六需求

<!-- MODULE REVIEW: AUDITED -->

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

## AUDITED：冻结 M1–M5

原source hashes已核对，现有CPU/CUDA/Apple三个consumer的packing、group次序、dtype和
checked IDs接口一致。21个原标准 tensor/arith/linalg/scf entry（3格式×linear/grouped×
3dtype + affine embedding×3dtype）已用锁定原compiler离线configure/compile成功；其
32个去重dispatch均为原LocalSize64，并沿用原precision程序的IEEE execution mode。
这只是编译/source资格，不是实际GPU PASS；不把scratch实验算成基础架构完成。

- **M1-F**：直接纳入该21-entry原序列。scf VM循环每次只持有当前K group；选中W临时为
  `[R,N,G]`，A为`[R,1,G]`，dot为`[R,1,N]`，F32累计为`[R,N]`，MX scale先以输出dtype
  materialize再乘dot。原`optimization_barrier`配合`flow.tensor.tie_shape`保持原dtype和
  tensor形状，已核对生成SPIR-V中U8 scale255的unsigned mask。embedding保留两次原typed
  mul/add舍入。不能全K或全expert bank展开。
- **M2-F**：私有七ProgramKind、experts、group与统一source/output dtype；A以二维、linear W/
  scales以E=1三维、grouped IDs以R一维的原native view承载同bytes；embedding保持二维bank。
  NativeProgram维持五输入上限及原scalar group，新增第五小型bytecode module。每invocation
  独立VM/input/private-output，原queue/fence/async callback/retirement与quota不改。
- **M3-F**：普通两入口接MlxQuantized。mixed affine仅冷追加已资格Cast，使A/scale/bias匹配
  schema输出dtype；checked IDs沿既有safe-index和checked-copy发布边界。workspace按
  group临时、两累计/dot/typed scale及必要转换预先有界；bank/input逐descriptor、index和
  native grid先检查。空embedding无需dispatch。源dtype/internalTensor必须真实计费。
- **M4-F**：原MLX源码/许可证/标准recipe/21 VM exports/新32原SPIR-V的manifest和vendor
  identity更新，复用CMake嵌入及许可证安装。OFF路线/其它owner原结果不动，不改公共治理。
- **M5-F**：本批全部三dtype/7入口、多group/实际packing、mixed affine、changed A/W/
  scales/bias/IDs、bad/duplicate IDs、两slot/lease/状态恢复/empty/close/cancel/OOM和
  bounds拒绝；新增DLL由原stage邻置。一producer、一不可变candidate集中build2/test1。

普通 API/生命周期不变；额外临时空间与原VM内部dispatch全部算成本，未知硬件以真能力拒绝。
旧上批通过域只在受影响共享接缝做必要delta，不重跑全路线/P1。模型16 actual geometry、
Air/三交付及父轮default/P1仍未验。owner现AUDITED重新绑定本卡，baseline/source范围明确；
同一原source/call误接最多一次具名纠正，否则列具体剩余域裁决，不无限优化或扩自研。

七问：原源/pin/MIT许可见本卡及SOURCE；普通主/反偏置链与失败形状已冻结；复用existing
Vulkan/ggml/IREE/mem/checked storage；P0 group空间和完整复制/执行成本有界；VM私有FFI
不进公开ABI；交付沿既有嵌入/许可证/DLL；用户只用原自然schema，最终能力/精度按真实域诊断。

M2-C具名成本修正：MLX bank实际可达1GiB；现有IREE边界每replay复制全部输入不能满足
不可变权重低成本重放。本批复用原native复制但拆成首次constant copy和每次dynamic copy，
只对Plan明确immutable constant及MLX生效；mutable banks仍每次收费。成功接受constant
GPU提交后才标记已初始化，取消/失败仍按原queue/drain退役。常量copy非benchmark-only优化；
普通Direct的`Argument(Tensor)`和Graph constants都适用。额外一native command只在cold
首轮，后续无bank copy、不重选Provider。预声明检查constant副本不变、动态输入/IDs变更、
两slot、取消/关闭/lease，原artifact/FP8资源路径不改变。

## 首次 Windows candidate 与唯一 fixture 纠正

`1ad27d6c8` 产品48步集中build PASS；`1fbd8ac7` GPU owner7/7、Direct8/9、Graph3/4。
MLX constants/rebinding、empty/OOM/cancel/close/零泄漏、MX8/源scale255分类及必要旧artifact/
FP8影响子集通过。仅两批量 ID在首个grouped坏索引后要求旧slot重新bind而失败；已跑域
没有数值或validation layer错误。原`vulkan_run.cc`保留每slot的terminal checked failure；
`gather_test.cc`既有合同明确旧slot保持OutOfRange、同Program新Invocation恢复。
唯一M5纠正只按此已有失败合同改fixture，不改变产品错误形状或数值门。
两失败ID执行一次delta；先前真实PASS域复用，不再跑Provider/旧域/全集或P1。
