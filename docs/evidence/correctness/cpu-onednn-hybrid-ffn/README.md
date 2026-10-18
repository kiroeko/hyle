# CPU oneDNN-labelled FFN：混合接缝诊断

已安装TC静态SDK产品`c2df0b8973a80b7140e69983c8e861668483a97c`；消费者HEAD9134c530，产品未改。
三个F32 explicit-accum、三个constant weights的普通四节点FFN实际为一项onednn-graph-region。
输入1x8，gate/up 8x8，down 8x8、每行平均输出；不发布中间值。
−80与−90最终结果匹配，−100/value1e38实得−0.000378350553，稳定F32 activation参考−0.000372044742，
差约1.7%。八个最终输出逐个检查，未计时、未做自研性能比较。

根因接缝源码是graph_provider_onednn的ActivationTask调用`simd::swiglu_f32_batch`；oneDNN承担
矩阵乘法，并非整个FFN数学都来自oneDNN。不能把这个结果记成原oneDNN logistic kernel失败，
也不能因Provider名称就把HYLE SwiGLU记成熟替代。初次down为8x1不匹配旧FFN资格，正确走
四个leaf且三点通过；outside-ffn-*保留这段独立正确事实，不能冒充完整region验证。

[diagnostic.zip](diagnostic.zip)保留两种消费形状、原源码、命令/日志与runtime/product身份。
这只签具体反例，不签产品修复、三交付、性能WIN或CPU完整；当前产品处于独立施工卡管理。


## 产品资格：CLOSED（NX1–3）

固定产品`80c9ab0075c667bd16255d43d46a21d46df234ac`仅删除旧通用FFN提案、私有artifact/workspace/
执行与HYLE SIMD callback；原oneDNN norm和MatMul/SIMD库保持。没有新数学、逐kernel执行器、
公开API或provider排序。普通Graph在必要边界组合已资格mature leaf/有效region，真正gate为ATen；
错误旧region不作性能基线，所有转换/packing/临时Tensor和融合损失纳入P0账单。

| 环境 | 实际测试 | 同一普通C++消费 |
|---|---|---|
| Windows | provider3/3；Graph4通过/2旧fixture skip；公开头1/1 | source shared/static、installed shared/static通过 |
| Linux GCC14 | provider3/3；Graph4通过/2旧fixture skip；公开头1/1 | source shared/static、installed shared/static通过 |

两skip是未改旧模型fixture，不计PASS。本批mature FFN八个最终输出、−80/−90/−100、large/Inf、
两个invocation/变化输入/独立旧结果和实际Completion已运行；expert-only不再获得错误FFN。
原norm region三节点保留、两线程独立queue/slot、warm HYLE metadata不增长、close后拒绝、Stats
资源归零通过，cold compile/invocation每次注入失败后live-bytes恢复并能重试。
原norm validation/alias/mapping/complete/context源码语义hash相同，相关旧资格可复用；未注入
native物理OOM/非coherent故障，不记故障注入PASS；没有新capture或SDK-OFF资格。

先前WindowsGraph“6/6”计数纠正为registered6、pass4/skip2；保留原XML。Status三元式初编译
失败仅test fixture，产品不变；ordinary第一次Direct caller workspace遗漏导致fast-fail，补齐三份
冷私有workspace后通过，原失败命令与returncode保留。没有重写失败为PASS或放宽数值。

[receipt.json](receipt.json)与[qualification.zip](qualification.zip)包含所有命令/失败/XML、同一
consumer/driver、1336输入manifest、norm source复用、源码/selector/cleanup/LOC。源码候选后未改。
root完成NX1–3与七门终审；P0，无新P1/SOTA/WIN。统一实现 **61,318行/249文件**，不含测试/vendor。
本批临时消费者已清理 **624,830,909 logical bytes**，保留SDK/producer/cache；硬链接不计实际空间回收。
CPU整体仍未完成，剩余成熟接缝与冻结自研去留另批推进；只有具名本批反例或相关依赖升级可重开。

qualification.zip SHA256 `47c91378e491334ef5ae1c156cd12536d998c0f4b4e89b5327b06533bb8c8462`。
