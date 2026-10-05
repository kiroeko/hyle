# CPU Cast/stable route：有限审计证据

[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-cast-stable-route.md)，baseline `b881d2788e0848404c536342b806d6642f3aefa7`。
以下tail-*记录为原已安装SDK的普通Direct/whole Graph尾部diagnostic；产品资格在后续单列，无性能/自研对比。

实际默认gate/Sigmoid与完整gate→MatMul均绑定ggml；Graph只有一个两节点region。
x=-90/value=1e38两者输出0，稳定F32 activation再乘value参考约-7.37461；value=Inf为NaN而参考-Inf。
x=-100同样失效，x=-80参考匹配。Sigmoid(-90)原返回0，F32参考8.19400869e-40。
6个门控组有4个具名反例；不是覆盖比例。完整输出/源码/命令见tail-*，不删除原日志。

初始SCOPED时Cast/Sigmoid未资格化；以下原程序、产品和候选记录依次更新该阶段，不以probe代签产品。


## 原版准入程序已固定

已锁LibTorch2.8.0 CPU，原DLL身份与原源码/命令/失败日志见original-*和original-manifest.json。
最终 **1,005,272断言**：25容量、993,220 Cast逐位、12,027三dtype Sigmoid。Cast全25组合，
完整F16/BF16/FP8源bits，F32特殊/random/tie上下邻值；独立数学decoder/nearest搜索/NaN位合同。
原to执行有限值转换，原Tensor clamp/where/bit-view/bitwise承担规范中的finite saturation/NaN mapping，
同dtype保持原bits，不写HYLE payload kernel。Sigmoid借原Double kernel后F32与storage舍入。

原fixture早期三个错误（缺header、Byte数量、低精度oracle少舍入）全保留；Byte容量修正后所有Cast
通过，不把无效测试构造误归因vendor。原程序资格不代签产品普通入口、资源、alias/strides或三交付。
该原程序witness完成后TC1–4进入AUDITED并实现；产品与交付资格仍须独立证明，Native/SIMD比较与其它路线继续冻结。


TC1补充：旧SDK默认ggml mixed Cast的NaN bits实际不匹配公开canonical。cast-nan-prior-*绑定原SDK0763fd3e产品；consumer HEAD为7eda8477（只有审计文档，当前产品草稿未编译）。原转换finite数学/region保留，仅NaN storage boundary列入本批有限FIX，所有产品门仍UNPROVEN。


## 当前产品资格（CANDIDATE）

不可变候选`c2df0b8973a80b7140e69983c8e861668483a97c`，与产品`689e9a1b`仅六个fixture不同。
WindowsProvider124/124与ggml58/58已通过；CPU首次158项114通过/7缺fixture跳过/37旧身份失败，
数值oracle未改。六个直接消费者白名单现仅增加精确ATen身份，保留原XML/失败位置与复用证明。
剩余Windows失败批重跑、Graph完整factor、Linux完整factor和两平台三交付尚未完成，不写PASS/CLOSED。


## 最终产品资格：CLOSED（仅TC1–4）

候选`c2df0b8973a80b7140e69983c8e861668483a97c`，算法/API/正常程序固定后未改产品。
Win/Linux各Native124/124、ggml58/58、CPU151通过/7旧fixture skip、Graph168通过/7旧fixture skip；
Linux公开头1/1。两平台source shared/static、installed shared/static同一ordinary consumer均通过。
完整CPU最初37身份失败原XML保留，六处精确ATen身份迁移后37/37通过；114通过/7skip以1329不变输入
和单调身份谓词复用。没有将Native任意fallback认作成熟，也没有降低数值或将skip签PASS。

五float storage全25 Cast对，F16/BF16/FP8全源bits、F32特殊/tie/random、rank0/8/empty/zero-odd stride；
同dtype原bits，不同dtypeRNE/finite FP8 sat/Inf/NaN，Cast全拒绝alias。stable Sigmoid原Double→F32→storage；
默认SiLU gate→MatMul的−90/−100、large-value与Inf输出实测正确，旧默认0/NaN反例已消除。
其他ggml有效region与具名模型原program保留；模型原内Sigmoid不继承泛型严格合同。本批不是整模型
或其它路线新资格。7个缺fixture的旧测试不计PASS；普通主/正交消费者实际运行。

复用原private native packing、checked mappings、实际Completion/异常/close及冷OOM/两线程资源回归。
native物理OOM/非coherent故障未注入，仅源码传播审计。原Double转换/clamp/where/bit-view/Byte packing、
所有原临时数据与热Tensor metadata计入managed成本；内部内存unknown、私有线程、不保证capture或
零热分配，SDK资格只到Win/Linux x86_64。feature-OFF按关闭branch/header/assembler完全相同复用MN
stub；只证明关闭adapter/TU/assembler，不声称全SDK关闭重建或移除Torch DLL。

root复核七门通过，tier P0；无新Native性能比较/WIN。统一实现LOC **62,010行/249文件**，不含tests/vendor。
消费者/探针已清理（除原探针硬链接外文件逻辑大小913,694,144 bytes），SDK/producer/cache保留。
实际命令、所有初次失败/XML/复用证明、同一consumer/驱动、1335输入manifest、selector、源码快照、
cleanup/LOC与复用关闭stub见[qualification.zip](qualification.zip)，摘要见[receipt.json](receipt.json)。
工件SHA256 `eb21ef006f32e3c5f0dc67677f0f7140133b74ab71f2b0a173cc954ca75eb21f`。

CPU全收口尚未完成；额外oneDNN-labelled完整FFN的小probe发现内部实际HYLE SIMD SwiGLU提前舍入，
−100/1e38最终输出差约1.7%，另owner有限卡处理，不将其写成原oneDNN kernel失败或本批已修复。
