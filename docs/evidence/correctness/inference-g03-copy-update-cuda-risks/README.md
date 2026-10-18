# G03 CUDA 原 update 的在途错误、取消与放弃

2026-10-14，按[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
固定candidate bcaec010b；[consumer](consumer.cc)、[实际结果](final.log)、[累计](receipt.json)、
[源码字节](source-identity.json)、[编排/host TU编译命令/原始日志](raw.zip)。WSL/GCC14.2，
actual RTX4070 / CUDA13.3，1024bytes原gate/memset/copy，最多4jobs串行。

四场景PASS：成功bytes49/原writer回接；request_cancel保CUDA原Unimplemented且
closed控制不返回writer；非终态cudaEventQuery观察OOM仍pending并保原guards，actual
wait后第一OOM保留、typed input/prefix/writer拒绝；未知时放弃两controls不提前释放
metadata/guard，原context实际drain与原abandoned-event reaper退休后全部metadata归零。
未知前提来自真实blocked原stream，decorator先调用实际cudaEventQuery，只有NotReady
才注入OOM；没有fault kernel/device poisoning，也不伪造原native工作或退休。

只在该validation source-parent给原cuda_backend.cc一个host TU启用已有HYLE_TESTING
observation hooks；没有新hook、产品源码修改或全局flag，EventSlot/layout不变。
编译日志只新增fixture与一个host cpp，已核其-DHYLE_TESTING=1；没有nvcc/kernel构建。
SDK另树HYLE_TESTING关闭，公开source/static/shared资格不消费此测试接口。

累计72.057秒（含约69.6秒原parent configure），两个阶段terminal0。
