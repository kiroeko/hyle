# H1 offload：WSL实际验收

产品候选：`7eaf67b8a496ee19b609533980cd69e7efb5aa49`。Windows为唯一产品真相。

- [source receipt](source-receipt.json)及[source测试](source-test.log)：实际5/5通过，包含普通
  CPU/CUDA Direct/Graph、组合、Direct权重迁移及资源/失败/生命周期contracts。
- [shared receipt](shared-receipt.json)：installed shared组合/Direct/contracts实际3/3通过。
  初次私有父项目HYLE_INSTALL OFF造成未生成SDK的失败记录保留，改私有配置ON后验收通过；
  不把早先cmake --install的进程exit0当完整SDK交付。
- [static receipt](static-receipt.json)：installed static组合/Direct/contracts实际3/3通过。
- [安装边界收据](seams-receipt.json)：shared/static四个changed安装头独立编译8/8、两种SDK真正
  改名搬迁后的组合消费者configure/build/run2/2通过。原prefix在搬迁期间不存在，未复制SDK。

[同步记录](sync.json)对应788,480字节delta；全部2,199个产品/消费者文件的Git blob身份已核对
当前候选。旧验证archive的生成版SOURCE-MANIFEST与Git原件不同，仅恢复其canonical内容。

[时间戳修复](timestamp-fix.json)只恢复未变native_bridge.h的最后Git内容变更时间；内容同时
匹配C3 baseline061b46700和当前候选。旧完整archive重置此header mtime，Ninja因其比旧CUDA
对象新而重复编译。恢复源文件mtime后由Ninja正常检查依赖/命令；没有touch对象或修改Ninja日志。
已完成对象保留，源码cfg已通过部分复用，源码实际测试仍对当前HYLE候选执行。

所有Linux重型阶段使用独立systemd unit，启动时实读memory.max=12GiB、memory.swap.max=0；
Windows另监控物理/提交余量16GiB，本轮private观察门12GiB，触门只停自己的Linux unit。
source cgroup记录峰值约8.7GiB且OOM计数0；不使用systemd结束摘要中的256KiB作为真实峰值。
relay exit0仅代表命令退出，必须有native完整阶段终态及实际测试receipt才能签完成；主动停止
用于时间戳修复的首轮relay没有完整终态，不算source验收PASS。
