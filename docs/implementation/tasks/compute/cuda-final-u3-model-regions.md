# CUDA U3 R1：当前模型片段、状态与 MTP 接缝

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- baseline：`c0150bc93ffa099ada45ac401bba15a36a9ec996`；Windows唯一产品真相。
- 协调owner：`compute_route_cuda`；实际FIX才绑定Torch CUDA／ggml CUDA／Graph ggml CUDA等owner。
  本scope不授予数学产品修改；现AUDITED旧卡不能替代下面有限审计。
- trigger：[U3 R1](cuda-final-u3-mature-bindings.md) 要求继承CPU当前完整需求。
  CUDA测试已include `cpu_c4_model_branches.h`却未接当前Pro0813／Flash0731完整DSpark及三sidecar
  runner；旧preview测试、原矩阵32项不能代签当前state／carry／完整head。
- 固定输入：共同目录／family-witness-index的八族49个region关联、官方mixed dtype、当前四新
  branch profile、Qwen Q3/Q4/Q6 named sidecar、两个Flash Q8/BF16 sidecar、本地Qwen3.5原版。
  同时保留全GGUF／non-GGUF基础合同；不另造模型画像或release。
- 主纵切：普通CUDA Direct／Graph完整program，权重／precision／真实内宽／cache／carry／
  confidence／Markov／三stage／完整vocabulary保留；少量token／驻留expert和head分片按原合同。
- 正交纵切：变化输入／IDs、状态反馈、两个独立slot、OOM／失败输出、close后accepted work保活。
- scope：共同program与当前CUDA admission逐step对账；成熟候选真实缺口或错误接线才能FIX。
  权重准备与热program分开，Provider原融合／资源管理不因HYLE调度需要拆坏。
- non-goals：自研CUDA数学冻结；不改CPU产品／公共语义或旧Vulkan BLOCKED；不写模型engine、
  loader／tokenizer／CPU+GPU模型分担、新通用Executor／验证框架或无需求大算子目录。
- resources：复用CUDA primary／stream／Mem及Compute/Graph Completion；actualstate反馈，失败output
  不消费；unknown private allocation／capture如实声明，cold binding与hot execution边界保持。
- tier：P0与correctness；缺口首次单成熟承载不造P1，当前重复域在U4同合同全成本有限裁决。
- budget：一次program／source／admission批量审计，固定有限FIX后一次批量接线／集中双平台验证，
  至多一批具名纠正。没有真实成熟承载则具名有限决策，不扩自研或无限换probe。
- predeclare：Windows／Linux原program两入口、当前dtype／物理格式、完整state/head、失败／OOM／
  repeated execution／capture限制及直接消费者；最终U3 immutable candidate集中source/static/shared、
  CUDA-only／mixed、offline／relocation／邻exe DLL，必须在U2/U3关闭前完成，不留U6首次接线。
  docs／closure／selector、统一Compute+Graph非测试LOC。未运行不签PASS。
- stop：原输入与当前CUDA map固定后AUDITED有限FIX才能写产品；所有联合资格前不签CLOSED。

## SCOPED 审计清单

| ID | 当前事实 | 本批审计 |
|---|---|---|
| A1 | C4当前model/sidecar program未在CUDA runner注册 | 复用相同数学／profile／callback，逐step typed admission与准备／热边界 |
| A2 | 旧D5八族完整region有CUDA runner但非本候选证明 | 按49关联去重映射当前test／code，保留未验证／缺口，不复读所有历史任务 |
| A3 | 原GGUF与新版FP8／FP4／E8M0来源/codec可能精度不同 | 对照已固定原carrier与当前role／rounding，禁止同文件Q标签或同op名代签 |
| A4 | 生命周期/独立slot/软件事实/SDK组成是路线合同 | 复用N1/N2机制证据，新增state／consumer适用项具名补齐，三交付联合验收 |

旧CPU已闭数学与性能只作为当前需求／源对照；本卡不重开CPU。原源研究以U1已有固定输入为准，
只为具体step反例做有界查证。审计后KEEP/FIX/BORROW/DEFER/REJECT写在本卡与小型coverage map。
