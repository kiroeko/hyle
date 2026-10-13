# G02 共同多 Queue 接缝阶段证据

本记录只签已实际运行的阶段证据，**不是 CLOSED 或整个 G02 完成**。活动卡见
[多 Queue batch](../../../docs/implementation/tasks/graph/inference-g02-multi-queue-batch.md)。
最终 API 为每 lane 的 Context、Queue、host workspace；删除原单 Queue 构造签名，迁移
现有消费者，保原 region/slot、F2 upload/plan、page permission 与 actual retirement。

Windows canonical 是产品输入，WSL只单向验证副本。内容比较后只复制变化文件，保依赖与
未改产品的时间戳；第一次复制18文件，per-lane staging产品轮再复制6文件。现有Linux
source-build/成熟依赖复用，后续消费者纠正只重编一个TU和链接，12 jobs。

实际证据见[receipt.json](receipt.json)与[原始日志](raw.zip)：

- Windows Graph机制/Executable（含新cold lane/精确Context Queue测试）、graph_batch与其
  public header通过；全Graph15头用本机CUDA SDK include通过。CPU factor树最初全头扫描
  缺CUDA include的实际失败保留，没有修改公共checker或将缺依赖写成语义NA。
- WSL RTX4070、锁定FI原数学：6组不同dtype/D/paged-ragged的两CUDA Context/两Queue，
  每region独立staging/device residency/metadata与两独立invocations，正常数学/旧结果/replan
  通过；同Context共享metadata额外回归通过。**两CUDA Context不是CPU＋GPU真数学证明**。
- 6个故障阶段通过：两Context的upload/plan部分prefix、first accepted region错误、unknown
  跨Queue前缀阻止后继，drain真实等待且保guards，失败不发布/不重用。原精确页append/F4
  old/new/old及真实第二request ragged重排也沿迁移接口通过。
- 原生前两次因global staging仍属第一Context而失败；per-lane staging后修正。一次replan
  断言错误按独立metadata已先更新一region的真实topology纠正；失败日志保留，不签为PASS。

首轮Bash/WSL命令包装未可靠传播native exit code；依据日志识别FAIL，随后用Python
subprocess直接记录实际returncode/elapsed并退出。最终完整consumer实际exit0，不由工具
显示的首轮exit0推断。脚本仅scratch编排，不增加产品/公共工具。

仍待本批三交付与适用sanitizer/最后P0及审查；G02后续还需CPU＋GPU准确分页和
Apple/Vulkan hybrid接口/数学/权限旅程。Graph与graph_batch仍AUDITED，goal保持active。
