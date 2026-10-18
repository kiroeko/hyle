# F5 Apple/Vulkan typed Direct与Graph控制器首批

AUDITED checkpoint，未签candidate/全部fault/全F5。actual Win RTX4070 Vulkan、Air Mac16,12
M4/MLX与M4/MoltenVK三profile：每份原48+新controller48矩阵oracle=96，原12+typed
controller12页oracle=24，合120/profile。八域paged/ragged、F16/BF16、causal/noncausal，
Direct/原native Graph、两个独立controller及显式layout更新均通过。四raw archives保存
common与三profile的准确身份/命令/inputs/history、原全部FAIL及累计预算，不覆盖旧证据。

公共Native MaterializedBatchInvocation cold create只编译固定common recipe的Direct。
graph::apple/vulkan::prepare_materialized_batch生成自己不可变fixed Graph，原native compile
后经private qualified create接同typed controller；public factory不接受任意caller Program。
actual Graph native owners合法AUDITED固定factory和直接类型/依赖边；原compiler、control/
effect准入及runtime未改。共用fixed F5 metadata traits glue不执行math/造Queue或fallback。

cold固定一个staging bank、两个device indices/mask banks，原F2 Program/invocation与原
native slot；prepare生成owned layout snapshot，选原bound bank之外的一份上传，重复未
bind prepare复用候选bank。旧native输入释放前不能原地覆写其mirror，原F1 try_write与
原F2 reset守实际权限；无unbounded grow或每run规划。bind同代Q/K/V/page frames和本
controller镜像，submit只原native guarded Program，F1 use保Q、actual K/V、metadata到
原Completion/OwnedResult退休。保两bank有费用，不伪称全部模式/capture或resident对称。

示例metadata1920bytes：staging有效1920、两个device mirrors3840、实际冷请求backing
含6块alignment slack合6138；本epoch uploaded1920，copy host waits Apple2/Vulkan1。
report单列这些事实，max_metadata_backing_bytes包括slack；wrapper budget与原opaque
native/private scratch/driver UNKNOWN合同保留，不把后者记0。两source pools的所有writer
先分配，再接不反持pool的recipe/allocator/budget owner后detach root。

private reuse_gate只成功idle原slot：原ResultSlot.begin/rollback拒old result/completion，
Vulkan还查原pending/observers/retiring/completion-owner/output buffer owners。可清无人
旧结果payload但不accept math；原普通readonly bind合同保持。failed/unknown/partial
upload不能提交混合generation或复用failed控制器，drain委托原native/F2/F1实际退休。
这里只签正常与列明边界；完整注入producer/unknown/cancel/accepted-prefix故障尚未验。

实际F16/BF16页journey old IDs2/3→new2/1→old，无page0，Qcap2/active1/GQA4:2/NaN
tail。原F2 clone1024+append512，whole K/V arena5120未复制；成功ready后F4完整目录
dirty page/layout替换与expected-base发布，clean block分享；独立uniform SDPA oracle
0.25/0.375/0.25。submitted期间free page4仍可取writer，used page2拒绝。typed controller
cold backing cap/metadata OOM、wrong Q/layout、mixed/未bind submit、old result阻止update、
重复prepare、close拒新工作、controller释放后原native result仍download，最后quota0。

common33项与九compute头独立TU通过，含新fixed traits header明确自含batch_transfer。
每profile六头（native batch两头、private bridges两头、native Graph两头）独立TU/Werror
通过，另一缺席route头可include。Win native owner/WX、Air原新batch-Werror通过；原
source/shared trees/cache复用，j4/j2/j1按available memory，test1，重型跨机器串行。
不是installed SDK/negative symbols/迁址/loader/san/P1证书；未运行旧算子全集或公共工具重写。

本goal四路线已有实际controller/页消费首批证据。仍待ragged实际append、完整producer/
error/unknown/alias/OOM/cancel/close风险、18适用SDK格/sanitizers/P0/LOC/guide/redteam/
legal final closure，以及用户最后的foundation缺口与单机engine代码量评估。
