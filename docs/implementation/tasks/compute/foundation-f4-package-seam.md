# F4 Compute façade 安装闭包接缝

状态：CANDIDATE，固定29d3c7a3e。新[compute_state活动卡](foundation-f4-state-versions.md)实现已固定，原Compute
公开产品新增state root是实际source/install/public发现合同变化；不是原F1–F3或native算法缺陷。

## Closure card

- baseline：f972101e4；原产品已签代码基线bc9a689f983bfa5bf9da9a30019ff58f033378f0。
- owners：compute、compute_backend_cpu/cuda、compute_provider_native/onednn、
  compute_route_cpu/cuda/apple/vulkan，仅Compute产品façade成员/安装闭包；源码/原公开头不改。
- trigger：HyleComponents的Compute roots加入compute_state；selector实际拒绝CLOSED→CLOSED
  product mutation，原因与plan保留在.scratch/foundation-f4/core-validation-plan.json。
  保持自然hyle::compute发现方式，不能改为新内部product、偷偷绕selector或重写公共工具。
- consumer：F4普通四route source/installed state share/clone/publish；主recurrent/conv/KV布局
  与正交资产集合，调用原Compute对象和新State API，同工具链C++。
- scope：有限F4-PKG-01，中央新root及三交付/新头/缺路线/迁址/loader；每个原factor的
  F1–F3与已有数学/执行/退休行为都按原已签证据保留，不重新审计/修改这些算法。
- non-goals：旧factor产品实现、全算子/全库回放、依赖升级、公共构建/selector/性能工具改写。
- 依赖：新state与原façade roots；无反向Compute→state边、无Graph循环。
- P0：新opt-in对象才分配；普通Tensor/Invocation不加字段/执行路径。façade新增链接/运行库
  按实际SDK验证，不把packaging-only签成新native吞吐/capture资格。
- 预算/机器：沿F4六FIX当前累计预算，configure300/build1800/test120/consumer180；Win CPU/
  CUDA/Vulkan、WSL CPU/CUDA、实际Air CPU/Apple/Vulkan；软件Vulkan只包。库按可用内存
  j4/j2/j1、test1、重型串行，原source cache与独立配置binary树，不因此卡重置任何预算。
- 预声明：中央组件/真实target links与factor授权；F4实际Compute façade source与迁址SDK
  static/shared、新头独立TU、未选route具名诊断、loader真实新state依赖；必要直接消费者。
  普通docs/closure/diff；registry previous-ref与status生成。旧factor cpp未改，旧算子、F1/F2/
  F3故障及性能不是新的测试根；新F4风险由F4卡验证。不因manifest保守选择扩大为全库run。
- stop：无法给合法授权/组件/包verdict则该接缝BLOCKED/INCONCLUSIVE；不伪造旧factor缺陷。

## 原已签范围

下面只保存新增接缝前registry的record/candidate/performance身份，不回写原terminal记录。
本轮source/public header没有这些owner的产品diff；它们的新状态只允许façade接缝工作。
old-scope.json与本卡同目录输入在
[原状态快照](../../../../evidence/architecture/foundation-f4-package-scope/old-scope.json)。
已完成独立SCOPED checkpoint b157d0963。此前未提交façade root修改；现在按本卡冻结接缝。

## 有限审计与七问

| FIX | 允许变化 | 验收 |
|---|---|---|
| F4-PKG-01 | Compute façade加入state root与新factor唯一归属；旧九owner源码/公开头无diff | 普通Compute发现新API、actual source/SDK新闭包/头/迁址/缺路线/loader，各route原native对象可组合 |

不改变旧factor直接依赖边/ABI/对象布局/算法/线程/metadata。只在Compute产品聚合选择新root，
新state依赖旧Compute，不存在反向边。header owner及source/static/shared符号仍由新factor。
中央manifest的保守全图测试选择保存理由；实际gate用F4 card的owners/new consumers/package
seams，不运行旧算子全集。新源码身份/旧owner无diff随候选收据核对；未经验证SDK不写PASS。

七问：锁/许可不变；主/正交完整state消费由F4卡覆盖；旧下层BORROW；P0只解释新增产品链接
和opt-in状态对象、无timed赢家；同工具链C++无新FFI；实际Compute façade三交付与新公共头
资格；指南明确State接口/原route选择/库依赖。故固定此一个FIX进入AUDITED，七门资格尚未签。

## Candidate 交付续记

2026-10-12：[Windows CPU+CUDA static](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-cuda-static/README.md)
实际PASS。固定29d3c7a3e，复用原MSVC/shipping/static/route树，只构建新state及直接Graph
route依赖targets；自然Compute/Graph安装后rename迁址，metadata无原source/build/staging
路径，普通消费者及三个独立公开头通过。消费者只在迁址prefix发现包，运行PATH只留
消费者目录/系统目录；缺Apple/Vulkan请求得到具名拒绝。原库产物与runtime hashes、实际
imports/完整stage原日志及时间见job/raw archive。其他installed格尚未因此代签；本卡继续
CANDIDATE，旧factor产品源码与原公开头未改变，F4终局风险/红队仍沿主卡。

同候选[Windows CPU+CUDA shared](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-cuda-shared/README.md)、
[CPU+Vulkan static](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-vulkan-static/README.md)、
[CPU+Vulkan shared](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-vulkan-shared/README.md)
亦全部实际PASS；至此Windows混合路线四个installed格完成，真实state DLL/其他运行库hash及
imports、受限PATH运行在各job/raw archive。native-only、WSL与Air对应格仍继续，未运行格
不由这四个PASS代签。

[Windows native-only Vulkan static](../../../../evidence/correctness/foundation-f4-native-sdk-windows-vulkan-static/README.md)
实际PASS：无CPU factor/数学回退，原Vulkan Direct/Graph完整state消费者运行，三个新头独立
编译；CPU/CUDA/Apple缺席请求明确拒绝。此时5/18个预声明installed格已验，动态native-only
及WSL/Air剩余格仍未代签。

[Windows native-only Vulkan shared](../../../../evidence/correctness/foundation-f4-native-sdk-windows-vulkan-shared/README.md)
实际PASS，迁址后state DLL及实际native运行通过、CPU/CUDA/Apple缺席请求拒绝；至此Windows
六个预声明installed格完成。WSL与实际Air余下12格按原候选继续，不扩全库算子测试。
