# Compute：计算语义、实现选择与资源边界

Compute 为 Direct 调用和 Graph 编译提供共同计算合同，保留 CPU、CUDA、Apple 与 Vulkan 的原生执行方式。
它不依赖 Graph，也不解释模型文件、生成策略或 serving。用法与限制见[用户 Compute 指南](../user/compute.md)。

## 语义与执行分开

公共语义描述 operation、全部端口的 dtype/encoding/shape/layout、alias、状态、副作用和资源要求。
Provider 把这个合同绑定成原生调用、完整 region 或原生扩展。设备能分配内存不表示某个 Provider
能计算任意格式；库名称也不能代替实际算子能力。

Direct 用户无需取得图编译器。Graph 在相同语义上组合片段、决定跨片段边界和资源计划，
Provider 继续拥有片段内部的融合、packing 和执行。避免把成熟 region 拆成逐 kernel 的虚调用链，
也避免为统一外观复制一套线程池、队列或 allocator。

## Provider 与资源所有权

接入有三种形状：直接调用原生 primitive；将 op/region 编译为 Provider 自有程序；
在 Provider 明确支持的扩展点内加入原生 kernel。三者可以共存，选择依据是完整计算的成本和能力。

Provider 自有 allocator、tensor、stream、workspace 或 capture 时，HYLE 保留这个执行岛：
边界完成合法 import 或显式 copy/repack，内部使用原生程序，输出由真实 owner 保活。
结果不必复制进 HYLE 影子 buffer；也不能修改第三方内部字段或伪造 buffer 来宣称零复制。

caller-owned storage 只通过实际支持的 typed interop 接入，核对 device、范围、编码、coherence、
borrow/ownership 和完成边界。CPU external pointer 可用，不能推出 CUDA、MLX 或 Vulkan 同样可用；
同一 GPU 上的不同 runtime 也不会自动共享 allocator 或 stream。

对 ggml，合法适配保有 tensor metadata、backend、常量和 invocation state，用原生 graph compute
执行已准备的单 op 或完整 region。其内部仍可能编码、分配或同步，应报告真实成本。
旧桥接中“简单节点成功而 varied-shape 图因伪造 buffer 崩溃”的教训是保留公开 import/allocate 边界，
而不是恢复旧 adapter 或把私有 kernel 函数当稳定接口。

## 冷路由

普通用户只选择可用设备和实际请求，不排列 Provider、不填写 timing profile。
实现规则由库维护，在 Context 创建和 prepare/compile 时匹配；submit 只调用已经绑定的制品。

匹配依赖三类只读事实：

| 事实 | 归属与作用 |
|---|---|
| `RouteEnvironment` | Context 捕获的 OS/ABI、设备、ISA/SM、features/limits、driver/runtime/SDK 与 Provider 身份 |
| workload / `ModelArtifactProfile` | 引擎已核对的模型/制品 revision、配置/manifest digest、phase 和 region 意图；不包含权重或 parser |
| `RouteSubject` | 本次 operation 全部端口，或真实 Graph region、canonical boundary、state/effects 与资源约束 |

模型画像可为空，普通调用仍能使用合法的通用基线。画像、字符串和 digest 由冷 owner 保活，
不借用调用栈；模型名或文件名不能代替真实端口检查。未知事实与明确的零值不同，
规则要求的事实缺失时不命中。region 意图不能替代 Provider 对真实 DAG 的检查。

选择先排除未交付或不满足语义、资源与设备条件的实现，再应用固定规则；Graph 还考虑完整计划的
覆盖、融合、搬运、等待和 workspace。稳定 Provider/implementation identity 用于诊断和缓存，
注册顺序、数组顺序和“规则更具体”均不代表更快。同优先级的冲突须明确处理，不能随机挑选。

只在接纳前、无副作用的能力或预算拒绝时，才可尝试预先声明的合法后备。创建制品或接纳工作后的
真实错误不吞掉、不换设备重算。匹配不到性能规则可使用合法成熟基线，不能把它称为测量赢家。

## 制品与执行状态

不可变代码、权重和 packing 可以共享；bindings、scratch、descriptor/command pool、可变库状态
及输出按原生并发要求独占或池化。同一槽未到真实终态，不得重绑、覆盖输出或回收其资源。
共同Operation/Graph invocation接纳后若失败，终态意味着该槽退休，需要重建槽；
不是错误一返回就可重试。接纳前拒绝可按合同修正输入再试，重建槽也不能修复device loss。

数值/物理格式、shape domain、所读环境事实、Provider/policy版本或相关常量身份改变后需要重新准备。
普通 payload、token值和域内实际尺寸仍是输入。跨制品 cache由消费者明确拥有，不使用裸地址、
仅模型名或进程全局隐式 cache 代替完整 identity，也不承诺跨编译器持久化。

普通prepared成功准备后的replay不再做shape推导、heap分配、JIT、tactic搜索或packing。
managed可承担已声明的metadata/array及cache/shape工作，不因此允许JIT/search/repack或隐藏copy/sync。
需要native planning的批次走独立layout准备，见下文。prepared/managed、capture与是否阻塞是不同轴，
不能以同一个“graph replay”名称掩盖不同成本。

## 存储、同步与生命周期

Mem 提供 allocation、mapping、view、quota 和寿命原语；Compute 解释 native resource及异步使用；
Graph 提供依赖和 liveness/alias事实。host metadata、payload、第三方 heap和driver内存各自计账，
未知私有内存不能记成零。释放使用原 owner、size和alignment。

CPU happens-before、CUDA event、Metal hazard/fence/event和Vulkan execution/memory dependency
保留各自语义。unified/shared内存不消除同步；必要的 host wait应显式报告。
只用于 host编码的临时数据可以按合同提前释放，设备仍引用的数据、参数和代码必须活到真实终态。

部分接纳后保留已提交前缀；事件记录失败不表示没有接纳GPU工作。取消停止后续工作，不等于抢占
既有kernel。停止接纳与显式drain分开，具体close是否等待服从该owner的API；共同Context的close只停止接纳。
未知完成状态不得变成可回收状态，
不能靠析构隐藏无界等待。资源权限和流送还需要下述约束。


## 资源权限与有界流送

只读租约可共享，独占写权限要求相应范围没有读者或在途使用。alias按物理范围而非wrapper地址判断；
成功写入后才发布ready只读内容。Direct/Graph绑定同时保有payload、索引/参数metadata、producer与native owner，
直到真实最后使用。引用归零与GPU退休是两个条件，不满足任一条件都不能回池。

一个重要反例是“事件记录失败”：前面的kernel可能已经接纳，错误返回不能触发正常释放。
资源必须留在仍能推进退休的owner中；若完成状态未知，不能凭Status非OK判断它已经结束。
这种保活也不能形成失败suffix永远持有producer的引用环。

批量byte copy冷准备固定域、范围/groups/slots与bytes上限。独立copy可重复读取source，
不能隐含destination重叠或批内依赖；有依赖时用明确阶段。全批预检后仍可能部分接纳，
需保留实际prefix与所有可能被引用的控制数据。copy保持物理bytes，不替代编码转换。

文件流送只处理已有File/Mapping范围与resource范围：

```text
文件 → 有界staging → native copy → group ready → 计算使用 → 回收
producer完成 → read lease → D2H/共享可见 → positional write → 完成
```

旧ready/use leases继续占slots和预算，容量满产生背压，不自动淘汰用户内容。
已成功组可消费，后面失败不把整批写成成功。direct IO、short progress及不能被逐操作抢占的文件fallback
沿原IO合同；取消不等于中断pread/pwrite。格式解析、文件命名、原子发布政策与模型cache属于引擎/IO。

## 集合发布不是多设备事务

状态schema声明有限opaque成员和资源约束；只读版本共享owner而不复制payload。
draft分享未改成员，显式clone将修改的成员，或替换为成功的out-of-place输出。不能修改base，
也不能把外部仍可写的native alias称为immutable。

```text
base → draft共享/显式clone → 新输出全部ready → expected-base检查 → 发布新只读版本
 └──────────────── 旧读者与native uses继续保活 ─────────────────┘
```

发布仅改变集合可见性，不回滚已发生的外部device写，也不表示文件已持久化。
冲突不改变current，不能按branch完成顺序自动选赢家；token绑定store、schema与generation避免ABA。
旧版本、draft、目录和copy槽有容量上限，clone按dirty bytes计费，不隐式复制整模型。
停止新更新/发布后，按本store合同显式退休自己的工作，不关闭用户Context。

这可承载推理分支，但checkpoint语义、接受前缀、RNG/token和模型策略归引擎。
recurrent算子有initial/final输出，不自动证明任意中间前缀可恢复。

## 分页计算为何需要独立准备尺度

Paged/ragged attention保持typed数学schema，不将单storage Tensor改成含混分页对象。
paged描述arena、page IDs、indptr、tail与真实active counts，ragged描述packed segments；
metadata、device mirror、plan与payload必须有相符generation和owner。
接纳前校验单调、边界、dtype/layout、overflow/GQA、空序列及causal相对位置，padding不能作为真实KV。

| 时机 | 工作与边界 |
|---|---|
| program准备 | 固定算法域、dtype/head/layout、capacity、workspace和slots，绑定实现 |
| layout准备 | 复制/验证host快照、native planning、metadata上传及必要host等待，产生本slot/generation的token |
| native提交 | 执行已准备批次，不再读取可变caller span、搜索Provider或隐藏新planning |

原生planner可能分配临时vectors或查询occupancy，这些是layout准备成本，不是无规划replay。
多个并发槽各有mutable plan/workspace；布局或失败退休改变后旧token失效。
只改page IDs能否复用计划取决于原生算法，不能按地址/尺寸猜测；固定batch和稳定地址也不足以保证capture。

Graph协调同slot的region与prepared token，不能在外部规划A后提交B。
只读attention不制造effect，KV append/scatter等真实写入才需要writer与顺序。
把paged数据gather为连续临时再执行dense数学可以是明确参考路线，但必须计copy/capacity/workspace，
不能把它当原生paged kernel。各后端的dtype/head域与capture能力仍按实际实现说明。

## 推理负载与数值边界

需求从真实权重角色、端口和跨调用状态出发，不能把文件名Q4/Q8当整模型dtype，也不能用小矩阵
覆盖完整推理。以下区分影响公共合同和Provider选择：

- storage、activation、accumulation、output、cache/state精度分别表达；FP8、GGUF、AWQ/GPTQ、
  NF4、MLX affine/MX等编码具有不同codes、scales、zero-point、group与舍入规则。
- attention包括mask、GQA、位置、空/变长序列和KV反馈；recurrent/causal conv的initial/final state
  与projection权重量化是不同边界。物理布局相同之前不能直接复制编码数据。
- MoE需保持路由tie、重复/偏斜expert IDs、grouped计算与combine顺序；动态IDs不等于重新选Provider。
- MTP/speculative需要可共享旧状态、显式候选更新与成功发布。接受前缀、token/RNG与模型分支策略归引擎；
  基础状态或分页能力存在，不代表已完成MTP引擎。
- 图像encoder/projector需要其实际算子、布局和资源链路；不能从文本投影算子推导多模态完整性。

### 中间值与反馈状态的反例

原生softplus若用`log(1 + exp(x))`，F32负尾的`1 + exp(x)`可能舍入为1，结果变成零。
融合region不单独公开中间gate，只说明它应按实际公开的attended输出和next state核对，
不等于放宽generic Softplus的数值合同。原始GDN primitive也不能自动代替外层Q/K归一化和动态gate计算。

连续状态验证必须让每个实现消费自己上轮输出的state，才能看出误差是否累积。
每轮重新注入oracle state只能验证单步，不能支持长期反馈的结论。

typed custom execution须说明端口、数值、layout、alias、workspace、副作用与失败；配置和原生代码
保活到终态。公开扩展不要求消费者包含`src/`私有头，也不用进程全局字符串注册替代对象局部owner。

## 依赖与性能取舍

公共语义核不拉入GPU SDK；原生runtime接入各有依赖和资源边界，普通产品由构建系统自动组装。
必要实现已在[构建图](build-system.md)中表达，不让用户为每个算子选择第三方库。

性能改动比较完整且等价的候选：同Provider直接调用的包装成本，与不同Provider的整体竞争力分别说明。
packing、数据变换、host submit、最终完成及资源峰值不能省略。收益不足时保留成熟实现；
不为替换单个leaf破坏更大region的融合和资源复用。
