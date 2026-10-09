# Vulkan V2：共同实际端口与完整状态链

<!-- MODULE REVIEW: AUDITED -->

baseline `9881ce1cf`；owners `compute_route_vulkan`、`graph_route_vulkan`、
`compute_provider_ggml_vulkan`。属于[父轮](vulkan-cpu-cuda-apple-parity-closure.md)。
共享 Compute/Graph、CPU/CUDA/Apple、ggml core 与公共验证仪器不重开。

## 审计与有限 FIX

需求只来自[共同目录](../../../architecture/model-compute-catalogue.md)、其 JSON 与现有
`cpu_c5_*`、`cpu_c4_*`、D5 主干/状态 consumer。559/80/34/45/104/八族不缩小；
沿其它路线已声明的完整 K/N、全词表 row 分块和显式 expert window 验收合同，报告
原 E 与执行 E；不把 window、冷准备或微型程序冒充完整巨大银行 GPU 驻留。
另保留三 IQ1 原 E 冷域与可用银行实跑任务，不以编译或静态 source 代签。

| ID | 已定位问题 | 有限修改 |
|---|---|---|
| M1 | 45 plain 中4个 F16/BF16 conv；公开 CausalConv1d 允许三 dtype，现 ggml hook 只 F32 | 沿既有 native Cast→原 ggml ssm_conv→原 typed Cast，F32 累加/输出舍入保留；无新数学 |
| M2 | 原 C5 common fixture guarded CPU/Apple，Vulkan 的旧 runner 无 cold preparation/invalid-index/replay phase | 复用共同数据和程序，补 Vulkan 测试承载与独立原 GPU/codec oracle；两普通入口、prepared constants、反馈/reset与坏 IDs |
| M3 | grouped 原 ggml_mul_mat_id 未绑定公共 F32 accumulation；原 ggml_prec_set_acc 明确支持此 op，Vulkan 原 map 读取 op_params[0] | 与现 MatMul 同样调用成熟公共 precision setter，拒绝设置失败；实际宽度/多token selected 差分资格 |
| M4 | 两旧官方 MoE 测试仍要求缺 hook 拒绝 | 改成原完整片段运行验收；新版 Pro0813/Flash0731/DSpark/MTP独立接入验收，不改模型源或数值门 |
| M5 | 新目录普通形状尚未实际运行 | 同 producer 一次集中矩阵，预审容量/feature/stride；具名额外接线缺陷先在本卡补明确有限 FIX，不无界实现 |

原 `ggml.c::ggml_prec_set_acc` 两 MAT op 同源接口；原 Vulkan grouped selector
`ggml_vk_get_mul_mat_mat_pipeline_map(...dst->op_params[0],true)`证明源参数相关。
M1 公共 schema 的 F32 accumulation 与 matching output dtype 已固定，复用原 casts 与
资源机制；所有转换、临时、managed/native 同步计完整成本。只改有限三 owner，
不更新依赖、不添加 HYLE shader、不恢复 self 数学研发/优化或扩张目录。

主纵切为 104 制品实际权重与完整 R02–R13、主干及适用 MTP/DSpark；正交纵切为
非GGUF plain/packed真实 dtype、存储/布局、changed inputs/weights/IDs、坏索引与资源。
Provider 保留内部融合；Graph 管全局边界；普通 C++ prepare/compile、不增加用户 flags。
PUBLIC API 不变，源码/static/shared仍同产品和原依赖闭包；三交付由父轮 V4 集中验证。

P0 只原算子/转换成本与实际资源账；本包不跑 P1、不写 WIN。审计/批接预算90分钟、
一次集中资格45分钟，每具名源/接线/fixture错误一次定向纠正，未完成明确标记。
一个 producer、build2/test1、重型串行；失败域分类后只定向 delta，文档不构建。

预声明验证：两入口559实际权重合同/80参数类/34 packed/45 plain/104身份、旧已验
完整8族分支及当前官方两个main/DSpark和命名sidecars；完整cold准备→hot重放，
原输入/中间dtype、finite/tie/尾部、state反馈/reset；容量拒绝/索引失败不发布、
资源/lease/cancel/OOM对具名变化执行必要delta。native已过微型包不重跑。
机器 Windows RTX4070真实 GPU；Air/WSL/三交付/default/P1仍父轮未验范围。

七问：原实现身份/许可保持；真实主/正交完整程序固定；下层和 checked/storage复用；
成本披露与原语义不变；无FFI/新provider vtable；原交付闭包；用户自然入口和具名限制。
本卡 AUDITED 授权 M1–M5，尚无本轮 runtime PASS，不签完整路线完成。

### M5-G：实际宽矩阵的原 grid 单位

共同plain `p033ae23672c4671a` N=17408/K=5120，现冷检查将8 split-K
乘到原始列数，误拒绝已审成熟矩阵。原 Vulkan MAT pipeline 的最小列 tile32，
实际 X=8*ceil(N/tile)，split reduce tile1024；原源码4419–4600/9208/9213/5804
可复现。只修这两个冷grid单位，保持8片F32 workspace和buffer/index上限；
matvec仍按不大于N的保守grid约束。不放宽数值，不改变原pipeline选择。

第一集中批次接入34 packed/45 plain及原官方四MoE完整片段；[原始证据](../../../../evidence/correctness/vulkan-parity-v2-catalogue/README.md)。
同版本共同fixture/误差门保持，native cold结果保留GPU常量；Graph同slot重放，
Direct复用不可变Program、按旧结果lease合同更新可变请求。原559/80/104及新增
完整branch余项仍属于本卡，未实跑不能由此批代签。

M2 下一矩阵已批接共同559 GGUF/104身份：只给测试增加外部Bank hook，现CPU/
CUDA/Apple分支不变；无CPU backend宏/链接/fallback。锁定原Vulkan 8 prototype
codec/dot为集成oracle，plain与三IQ1扩展用独立原解码/舍入方程，原K/N与全部
head行分块/显式expert window保持。旧80项两入口通过域复用，新矩阵尚未运行。

Windows 第一批 `6ed2cdefa` Direct80/80、Graph80/80 PASS，无validation layer错；
34 packed/45 plain实际合同及原四官方MoE完整程序已资格。M1/冷宽grid与cold
preparation普通入口通过；M3普通GGUF grouped精度绑定随后在559矩阵签运行，
不借上述IREE grouped PASS冒充。剩余559/80/104及新版状态/sidecars继续本卡。
当前整体约38%；统一非测试Compute+Graph62,412行/267文件。

GGUF首矩阵 `a3c3567f` 在原GPU oracle初始化处停止，尚无case PASS。
锁定Vulkan运行时有原instance隔离补丁，直接vendor全局init没有激活owner而assert。
只修测试oracle：创建既有NativeDevice、使用其backend和作用域包围原native调用/
释放；不改runtime/产品/数学/门。定向原559/104矩阵一次，原80项不重跑。
