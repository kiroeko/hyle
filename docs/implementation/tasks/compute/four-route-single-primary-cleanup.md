# 四路线：每个完整条件一个首选实现

<!-- MODULE REVIEW: CLOSED -->

baseline `5f26ad52`，用户明确要求CPU/CUDA/Apple MLX-Metal/Vulkan统一清理：每种
完整条件除默认/必要兜底，大部分只留一家被认为best的实现。当前[Vulkan父轮](vulkan-cpu-cuda-apple-parity-closure.md)
纳入此有限终审；不恢复旧goal/队列/百分比，不凭本卡自动重开其它CLOSED owner。

条件包含平台/硬件与版本、模型/制品/phase/region、实际端口encoding/dtype/layout/shape、
数值/资源/完成语义；“同算子名”不等于同条件。每条件只有一个固定primary，普通用户
不排序、不提供timing；default/fallback须命名真实用途，不能让退役候选继续轮流竞争。

有限审计：只核对现行四route默认/规则表及其中引用的实现，与各当前terminal已有裁决
交叉检查。列出KEEP_PRIMARY、KEEP_NAMED_FALLBACK、RETIRE_UNUSED/LOSING、UNIQUE。
不扩算子/性能研究，不重跑已过矩阵。SIMD本身、存储、索引与资源机制不是算子清退对象。
发现具体重复/闲置或不合理多primary，先具名定界和绑定该owner，再AUDITED有限FIX。
删除后的引用/默认/资格条件与必要consumer一起迁移，不因Provider其它域有用删整库。

审计预算20分钟、具名修改/一次必要验证20分钟；源/同合同结果清楚就直接按成熟默认
快速裁决，不优化挽留自研。若唯一能力拟撤回则列具体取舍，不盲删仍承诺的能力。
本卡不签新的WIN、全硬件SOTA或尚未执行的gate。初审只docs/closure/selector/LOC。

## 当前源码核对与有限裁决

[逐路线用途和当前源码SHA](../../../../evidence/correctness/four-route-single-primary/audit.json)
核对现行规则、冷选择和物理实现；CPU43个、CUDA23个完整资格请求均恰有一个
chosen primary，没有并列赢家。未测条件明确采用ENGINEERING_DEFAULT，不称实测best。

| 路线 | 固定首选与必要保留 | 清理事实／本轮结论 |
|---|---|---|
| CPU | 按端口/语义选择oneDNN、ggml、ATen；43精确请求可覆盖默认。ATen的稳定特殊值/odd-byte布局、GGUF原codec、Copy/RMS具名资源能力仍有用途 | F5已物理退役14种冗余数学；剩余20种有资格用途＋独有Copy。当前dtype分支已裁剪，不把旧35种当现状 |
| CUDA | 23精确叶规则；其余按plain/packed/稳定函数/索引/attention分别固定成熟库。Graph完整region优先，managed/低对齐/stride/严格特殊值是具名能力 | 已物理删除79个重复dense实例；保留的generic共享体同时服务真实胜出或独有域。它不是另一家热竞争者，不重复计作79种算子 |
| MLX/Metal | GGUF/固定模型region为ggml Metal，普通/MLX packed为MLX；四实测交集现有默认已是ggml。Q4_K exact-row改走唯一合格MLX source程序 | 无多余自研Metal/GEMM/GDN/Hadamard kernel待删除；FP4保真、codec来源组合和checked发布是必要接缝，不因名字HYLE误删 |
| Vulkan | IREE具名packed/precision优先，ggml合格region随后；未满足原512-lane/grid/16-bit条件时才进入128-lane具名能力。整数/Bool/odd-byte存储、完整IEEE/tail、空reduce、checked索引/重复更新和wide scan各有明确用途 | 初表“兼容”已细化到真实条件，未把这些自研标成WIN。当前未发现能够整体删除而仍由已资格成熟路径完整接替的剩余源族 |

此处是**本轮有界源码清理审计完成**，不是新删除提交或新的性能证书。没有为制造
删除数量撤回仍有consumer/标准需求的能力，也没有重新启动CPU/CUDA/Apple的P1。
本轮新增物理删除数为0；历史删除不算新进度。跨域有用的Provider继续保留，同一
普通条件只绑定一个执行程序；源码中必要的另一域实现不等于同条件多primary。

Vulkan平台/三交付仍由V4执行，不能由本审计替代。以后具体新替代/真实重复反例
才定向重开owner并删除；禁止用“兜底”空理由扩张self或保留已裁决输家。

验证：closure57组件/schema4、docs876文件/4158链接、diff检查通过；selector只选
docs/closure，132项有理由跳过，没有重建或P1。统一脚本Compute＋Graph为
**62,483行／268文件**，不含tests/vendor/bench，当前产品源码未变。
