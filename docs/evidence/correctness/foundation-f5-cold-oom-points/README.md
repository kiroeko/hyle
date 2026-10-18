# F5 CPU controller cold metadata逐位置OOM

baseline5581cacd3，AUDITED原F5风险OOM-POINT-01。没有产品/API改动，只在现有实际
CPU Direct/Graph consumer的冷create路径装饰caller allocator。成功基线测实际分配次数
并限制<=64，然后逐位置仅一次失败，核ResourceExhausted与所有已分配metadata释放。
单线程控制，不把原thread-compatible API当成可并发控制对象。

Windows MSVC19.50 Release/shipping/static/WERROR与WSL GCC14.2 Release/shipping/static
各Direct14/Graph45位置全部通过，共118。原正常32矩阵oracle、精确页/F4 old-new-old、
五个Gather资格图与ragged多请求append照原实际consumer通过。provider内部不经这个
allocator的原生私有allocation没有由本测试覆盖，不宣称所有OOM完成。

全部沿原Win `graph-windows` 与WSL `foundation-f5-materialized-cpu` 日志root、原配置与
构建树累计，configure300/build1800/consumer180，不重新取预算，也没有占用CUDA故障门
一次追加预算。重型跨机器串行、库按可用内存j4/j2/j1、consumer1，未回放旧全算子或全库。
完整失败/输入历史/stage命令/身份/hash/累计见[receipt.json](receipt.json)及对应raw.zip。

selector只选compute_batch测试影响；实际新fixture由这两个真实Direct/Graph消费者执行，
公头/产品/默认common33项源未变，复用原匹配证据。docs/closure/diff另检查。
native partial/producer/error/suffix及其他OOM、D256、SDK/san/P0/guide/终审仍未因此签证，
compute_batch/graph_batch保持AUDITED，完整goal没有完成。
