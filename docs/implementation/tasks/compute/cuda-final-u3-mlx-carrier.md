# CUDA U3 N2：原 MLX packed 语义的 GPU 接线

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- owner：既有 `compute_provider_torch_cuda`；不建立重复 runtime 或新数学 Provider。
- baseline：`bf70470a4a8501e94fb4f1e2102680e1a79fd850`，Windows canonical。
- trigger：[U3 N2](cuda-final-u3-mature-bindings.md) 的16个原 affine4／MX4／MX8 计算合同在 CUDA 无 hook。
  需求继承共同模型目录，不重新画像；旧CPU资格只证明合同，不签GPU通过。
- 主纵切：普通CUDA Direct／Graph，原packed projection／selected expert／完整embedding/head内宽，
  原BF16／F16／F32边界、F32 accumulation、E8M0与group缩放顺序。
- 正交纵切：改变payload／checked ids、两slot、失败输出与accepted-prefix保活、close／teardown。
- scope：用原MLX公式和已资格CUDA ATen库执行；cold shifts／FP4 lookup常量，原metadata／stream／
  context／异常／Completion复用N1；SDK不新增。原CUDA融合kernel必须先匹配精度，不由名称外推资格。
- non-goals：不改CPU产品／公共数学schema、不扩自研CUDA／SIMD、不引入模型engine／CPU fallback、
  不把成熟库改成逐kernel万能执行器，不改公共构建／治理／性能工具。
- public freeze：既有 `MlxQuantized` 三种operation，普通用户只选CUDA；Provider identity仍为
  原mature tensor runtime。原F32 accumulation与scale dtype行为保持，输出类型由既有infer决定。
- resources：shared HYLE原CUDA stream／primary context；managed private ATen allocation；每invocation
  私有result至terminal，独立slot可并行；严格private quota与external capture不承诺。
  索引检查先于原index_select，必要D2H／同步完整计入；无热编译／搜索／参数重打包。
- deps：既有Compute／CUDA backend／Ops／Error／Mem及原Torch CUDA SDK；新recipe由实际owner登记
  pin／hash／license。CPU recipe文件保持不动。
- P0：按K group展开选中expert，embedding只展开选中row；不展开全float expert库。
  GPU张量metadata／临时值及index同步是完整成本，不宣称原CUDA融合kernel的成本或SOTA。
- budget：一次原源／dtype／scale／资源审计，一批产品接线，一次稳定候选双平台集中correctness；
  至多一批具名接缝纠正。原成熟承载无法兑现合同才具名裁决，不另写数学或无界实验。
- validation：16实际合同×Direct／Graph，原物理字节与precision、重复／两slot／坏IDs／恢复、
  生命周期／原OOM／资源及capture拒绝；headers／真实消费者。N1／N2／R1的最终U3 candidate集中
  source/static/shared、CUDA-only／mixed、offline／relocation／DLL邻exe，必须在U2／U3关闭前完成。
  单载体只P0，不制造P1竞争；未运行不签PASS。docs／closure／selector及统一LOC。
- stop：上述有限接线与适用资格后进入候选／联合验收；本scope本身不授权产品实现。

## 有界审计输入

原MLX `2b2bb49790158ecd008d1685c94e3ef119792a00` archive SHA
`486e8b0626ab6bbc0a905bdcc47aa7db8b9a258a958ffefd4898aacaf1f7d396`。
U1原71必要文件已经固定；本次仅补原CUDA device／quantized header，archive SHA再核对，未运行或改数学。

当前共同schema明确原CPU packed公式／F32 accumulation、source dtype与group order。
原CUDA `qmm_naive.cuh`把dequant值存为输入Element类型的shared tile；BF16调用先舍入解码权重，
与当前完整F32公式不是相同内部精度。`gemm_sm70.cuh`的F32分支是原UniversalFMA，非TF32；
但F32改型仍须处理source scale/bias、MX先cast scale到activation dtype及极值，再取得完整资格。
`qqmm.cpp`还可能先量化activation，不能用这个路径替代当前未声明A8转换的qmm。

因此本批先承载原MLX逐group公式于CUDA ATen，不声称原CUDA融合已经接入或更快；常量factory／
device／checked ids是接缝，代码算术／rounding保持。原GPU融合候选留待具体等价合同资格，
不通过调整误差门、撤公共precision或省检查来接线。

## 待固定有限 FIX

1. 原MLX GPU tensor recipe的source identity／只适配device及cold常量；CPU源不改。
2. 现mature CUDA carrier绑定Mlx三种operation、overflow／checked indices／dtype／错误与保活。
3. 16原合同两入口及小型精度／失败／资源／生命周期批量资格；三交付沿上述联合批次。

这三项仍须下一提交AUDITED并将当前owner record绑定本卡后，才允许产品施工。
