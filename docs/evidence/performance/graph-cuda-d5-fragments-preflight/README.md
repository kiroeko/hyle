# CUDA D5：A/A 未资格化，未裁决产品性能

**工作负载 A/A FAIL；正式产品 P1 未运行，聚合性能 INCONCLUSIVE。**
SDK 产品为 `9273bc71`，最新消费者为 `82229660`；见 [汇总](summary.json)。
没有将 A/A 的偏差记成 Graph/Direct 产品性能失败，没有增加 n 或放宽5%门限。

## 实际执行

首次配置/编译完成，但 validator 读取 CMake 生成的 Windows 路径 JSON 时失败。
消费者的 A/A ready guard 阻止产品采样；见 [首次 validator](attempt-1/validators/cuda-workload-aa.log)。
仅规范化 benchmark 的路径分隔符后，固定新候选，重新进入有限批次。

第二次已校验 SDK 的393个相关文件、Qwen实际数据、recipe、三个原CUDA安装DLL和引用源码。
Graph/Graph、Direct/Direct 各覆盖执行双顺序×分配双顺序，共8个进程，全部正确执行完整消费者。
caller CPU2、parent mask0x14、above-normal、SM89、driver API13040/runtime API13040/toolkit13030
均经子进程 telemetry 核对。两侧输出与完成状态仍通过；原始数据见
[A/A 收据](attempt-2/d5-cuda-workload-aa.json)及 [8个进程日志](attempt-2/validators/cuda-workload-aa.log)。

同实现比率出现超过10%的粗不稳定，若干执行/分配 factor 的中位偏差也超过5%。
benchmark 内部置信区间同样宽；不能根据这些点值判断哪种产品路线更快。
A/A ready marker 未写入，正式产品首进程在计时前被拒绝，**P1样本数为0**。
目录名 `raw/n-15` 是 runner 的预建目录，不代表已完成15个样本。

## 有界环境复核

全部HYLE benchmark进程结束后，三次GPU观测仍为43%、42%、42%，P0、38°C、SM2835MHz，见
[环境观测](attempt-2/environment-followup.json)与[进程检查](attempt-2/environment-process-check.json)。
这证明当前还有非本批测试的GPU负载，
很可能影响测量；没有反向断言其唯一解释此前全部偏差，也没有归罪于某一个桌面程序。
源码预审同时确认两侧都计入最终 wait/settle；没有遗漏 Direct drain 或为某侧延迟回收。

本轮在这里停止采样，不调大样本、延长 epoch、改电源计划、提权或关闭用户程序。
恢复条件是出现本机GPU空闲窗口后，按原profile、完成边界、5%门和n15→31预算重新资格化；
若空闲时相同工作负载仍不合格，再按具体机制另行定界，不能无限重测。
期间推进其它路线的剩余域或交付，不阻止已有正确性收据继续有效。

原合同、CMake/编译命令与两个产物SHA256随每次尝试归档；[文件哈希](files.json)保留原数据身份。
本批不改变152/176正确性账本，不签CUDA A6或D6。

已删除旧消费者和第一次失败构建目录，释放1,197,492,272字节（约1.12GiB），见
[清理记录](cleanup.json)；保留唯一最新`C:/hv/d5p2`与既有A5静态SDK供后续工作使用。
