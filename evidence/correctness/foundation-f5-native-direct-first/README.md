# F5 HYLE native Direct 首批

AUDITED implementation checkpoint，未签candidate/Graph/全F5。WSL GCC14.2/NVCC13.3/RTX4070，
原锁FlashInfer0.6.18/69ff11fc；普通hyle::compute Direct消费者，仅新batch/原FI与必要
直接库产物，库j4/j2/j1、test1，原同配置source-build树复用。

paged/ragged各F16 head64短KV和BF16 head128长KV四个域、各三layout episode，对独立
double dense oracle共12次数值通过。原F2 metadata pinned上传→native planning H2D及原
Prefill/split-KV/merge→run；query拓扑变化update重plan，稳定indptr/仅IDs或version变化
更新device mirror并复用原plan，不在run隐式规划。host snapshot→pinned staging额外bytes
与host wait明确计账，原planner临时vectors/occupancy/private UNKNOWN和capture=false保留。

旧output阻止overwrite；所有invocation控件/Program释放后原result可read且原F1 native
read-use接纳，结果占用冷slot容量直到实际domain最后释放；QuotaAllocator最终归零。
fixed-block adapter冷detach_pool_root复用F1 adopt的domain生存规则，消除pool root与
resource-owner回指环，不改变常规F1 destructor/close行为，不增默认Tensor字段或新runtime。

原两次Direct FAIL（把system snapshot直接作为CUDA F2源/错误GPU tensor host假设）、
update lambda返回类型编译FAIL、原commands/输入历史与累计budget完整封存；修正用原
Context的pinned staging/resource contract，没有修改旧CUDA copy算法或写新kernel。
不代签cancel/unknown/producer全部风险、四路线显式reference/append/Graph/三交付/san/P0/终审。
