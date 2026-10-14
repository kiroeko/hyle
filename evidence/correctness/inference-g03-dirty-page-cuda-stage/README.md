# G03 GP01/GP02 CUDA 公开入口短资格

固定product `b42832de3ddda21bdc3e98ad4cc87e73622cbba4`；WSL2/Ubuntu24.04、GCC14、
CUDA13.3、RTX4070 SM89、driver617.14。actual身份、source/SDK命令、SHA与累计见
[receipt](receipt.json)，全部日志/consumer/runner与四owner LOC在[raw](raw.zip)。

只复用原WSL source/两个SDK树，定向common/batch/CUDA backend targets，四jobs。
source17steps、static15/shared18steps，无model算子或nvcc kernels重建。实际运行每次
低于1秒、应用payload<128KiB；原Windows重型GPU门未启动，不外推其资格。

[source64](source-final.log)、[relocated static64](static-final.log)、
[relocated shared64](shared-final.log)全部通过：1024 dirty/16384 page、原generation/
untouched history逐byte oracle、共享reader拒绝不消费、retained submission不返writer。
真实cudaLaunchHostFunc checkpoint与原cuda::submit Completion作为producer接F2；
checkpoint未释放时source use=1、无ready dirty output，实际退休后取回原writer并publish。
download只作独立oracle，不算实现把历史从GPU搬回host。

两SDK复制完全不变consumer，只find_package(Hyle COMPONENTS compute)、只链
hyle::compute；各三standalone headers（batch_transfer/paged_arena/backends-cuda）与
compdb核实relocated include，未从product source借头。源码fixture只为避免全闭包构建
显式链接相应factor，不把它冒充普通SDK产品入口。

没有CUDA math/完整pending producer→attention、driver/shader san、private pool或token/s
声明；这里只签两新公开机制的实际CUDA消费。原F2 CUDA aggregate initializer编译warning
保日志，零初始化语义未改，不由本任务重开backend。全部产品文件与候选SHA一致。
