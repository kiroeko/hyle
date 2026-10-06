# CPU原库Transformer/矩阵接入：审计记录

产品baseline7a6d3535，原程序consumer HEAD392f3e91，LibTorch2.8 CPU DLL锁定。
同一witness最终212个独立断言通过：rank8广播MatMul与K0、Bool非零mask/log-softmax/无有效行NaN、
长I64正/负位置RoPE两种pair layout与未旋转tail、GQA/Dk-Dv/bottom-right causal与zero-key attention；
F32/F16/BF16原F32转换、matrix/Softmax及RMSNorm/LayerNorm原kernel亦有数值断言。

原fused SDPA全mask输出0，公共合同要求NaN；不得直接代签。候选采用原matmul→scale/mask→
softmax→value Tensor math程序，零keys用原full_like NaN；所有score/转换/GQA repeat/axis临时计入。
RoPE是原Double sin/cos/stack/cat/clone/slice_copy Tensor程序，I64先按原Double转换，tail原bits保持。
矩阵与norm不做HYLE payload数学，Bool storage先eq(0)按非零定义，不能reinterpret异常Bool位。

首次oracle把visible +Inf漏判为无效行，212最终witness纠正并保留失败日志，不记成原库数学失败。
[original.zip](original.zip)保存所有源码/命令/失败/DDL身份；这仅原程序实际witness，不签产品
stride/alias/OOM/普通入口/三交付、性能WIN或CPU整体完成。


## 产品资格：CLOSED（TM1–4）

固定产品`3fc528dae5a25b7ac3f0bcc9606b112ee23b7c6d`，源/API/布局/成本候选后不变。
原matrix/Softmax/log、Double RoPE、attention math和norm均为原ATen kernel/Tensor程序，复用既有
对齐Byte packing/private lane/checked mapping与core lifecycle，没有HYLE新payload数学。
布尔位以非零解释，I32/I64原载体不截断；Rope clone保留tail bits。SDPA原fused allmask=0不代签
公共NaN合同，math完整程序保留F32 score→mask/causal→Softmax→value和zero-key NaN边界。

| 环境 | 实际测试 | 同一public C++消费 |
|---|---|---|
| Windows | Provider134/134；CPU12/1skip；Graph33/6skip；头1/1 | source shared/static、installed shared/static通过 |
| Linux GCC14 | Provider134/134；CPU12/1skip；Graph33/6skip；头1/1 | source shared/static、installed shared/static通过 |

所有skip是未改旧model fixture，不计PASS。新增10批量tests三storage/rank8/broadcast/transposed/odd
stride/K0、Softmax三mask/log/非末轴/empty/specialrows、长/负I64/prefix/两layout/tail NaN bits与empty
RoPE、masked GQA两layout/Dk-Dv/causal/zero keys、norm浮点affine/rank8、alias/两线程独立slot/close
及masked attention逐点cold HYLE OOM live-byte回滚/重试/资源归零均实际运行。
ordinary消费者F32默认ATen域走四族+norm、Direct和op→Copy Graph下游、变化输入/真实wait/close；
三storage数学由同factor实际资格，不虚写所有模型/所有SDK平台已推理。RoPE/RMS/LayerNorm alias
依据all-input packing先于output写入源码审计，Softmax代表alias实测。native私有OOM/noncoherent
故障未注入，仅源码传播；没有新capture/zero-hot-allocation或精确内存上界承诺。

所有Double/F32/storage转换、GQA repeat、score/mask、axis物化、原temporary与metadata热分配收费；
仍为managed/unknown/private threads，仅已锁Windows/Linux x64 SDK。已有有效ggml/oneDNN区域
源码保持；普通prepare冷绑定，不重新搜索/timing。没有Native/SIMD性能裁决或P1/WIN。
原212断言只作审计witness，初次oracle漏visible Inf以及Windows test API/type失败全部保留；
未降低数值门、未改产品算法或把skip记PASS。关闭SDK分支与header/assembler保持，旧stub复用
不冒充完整SDK-OFF重建。root已复核TM1–4与七门。

[receipt.json](receipt.json)、[qualification.zip](qualification.zip)保存所有执行/失败/XML、同一
consumer/driver、1337输入hash、源码快照/selected-skipped/cleanup/LOC。统一实现 **61,493/249文件**，
不含tests/vendor。本批消费者清理 **625,657,864 logical bytes**，SDK/producer/cache保留，硬链接
不算实际回收。CPU整goal仍有剩余资源语义/冻结自研有限裁决，不以本卡CLOSED代签。

qualification.zip SHA256 `c653d281d96fede0161f2c023d7365030f4c32f6a1aad6d59d55aa59a2c2f08f`。
