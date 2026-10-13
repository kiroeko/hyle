# Native page 性能消费者预校验

冻结产品 `520027a57`，actual Air Mac16,12/M4。两 route 的候选与对照在三组有限
F16/page16/GQA8/2 消费中均通过 independent double 绝对误差2e-3、nonfinite 与逐确切页
readonly 检查；这轮只执行 HYLE_BENCH_VERIFY_ONLY，未计时或获得 P1 verdict。

两边复用原共同 slot controller、同一 owned layout/I32 table、固定 staging/两 mirror
banks、F2 upload 和真实 F1 页 guard、两个 slots。每个被测 episode 都明确 prepare
原 native Program/Invocation、上传页表、bind/submit/wait 并返回和释放同一 packed
结果；当前冷准备与 native replacement 的成本不会移出计时。peer 每 request 只 gather
一次物理 K/V 历史再运行独立原数学，candidate 直接页地址读取；不以每 Q/head 物化
作对照，也不使用不同精度数学。

Apple peer 调用原 fast::scaled_dot_product_attention(force_fused=true)，避免原默认
D256 prefill 选择 unfused 路径造成数学身份不同；candidate 原 vector/NAX/dsplit 的
公式与 reduction 保留。Vulkan peer 从 pristine 锁定 ggml Git source 生成独立原
FullF32 scalar SPIR-V，原 mask/split-K/output adapters 保留；只在前面增加物理 page
gather。receipt/raw 保存实际 shader 身份与 consumers。

累计55.530秒，Vulkan两次脚本配置 FAIL（peer cache 没有 .git、错误使用 glslc 解析
glslangValidator 参数）及修正后的 PASS 全部保留；没有为此修改产品/公共工具或
依赖缓存。Windows 新候选定向编译、正式不可变合同 P1、G02 联合分页及整个 goal
仍未完成。正式合同使用原16 A/A、controls、n15/31、5%非劣门，不从本预校验推导性能。
