# F3 P0成本结构核对

本页由[活动卡](foundation-f3-file-resource-streaming.md)纳入，只核对原F3-01至08的成本结构。
它不代签尚未完成的交付与最终红队，也没有P1/P2计时或吞吐赢家承诺。

## 固定容量与复杂度

记请求上限Q、每请求组上限G、每组范围上限R、固定槽K、每槽stage字节S、已准备资源数T；
L=QGR是保留记录的最大范围数。默认Q=4、G=16、R=16、K=2、S=1MiB。
源码位置为`src/compute_stream/file_stream.cc`的`check_options`、`FileStreamAccess::create`、
`existing_accounting`、`backing_usage`、`interval_hazards`、`admit`和`kick_locked`。

- 冷创建固定Q个request、QG个group、L个range、K个slot、GR个admission、2L个interval和L+1个
  backing目录，乘法与stage合计都先检查溢出；这些数组不会按文件或模型大小增长。
- 数据stage容量为KS，active backing预算将该stage与全部保留的外部F1 backing按原domain identity
  去重后计费；同一backing的多个view不重复收费。旧ReadyGroup、导出ReadLease与未退休producer
  继续占记录/bytes，不能用collect或丢弃ticket冒充可复用。
- 接纳在共同锁内核对原范围、alias、设备、描述与padding。保留范围扫描O(L)，排序/扫描
  interval与backing为O(L log L)；每个新范围最多检查T个已准备resource和K个stage，
  因而接纳上界O(L log L + GR(T+K))。不是按整个文件长度扫描。
- 分块执行C个实际chunk，每个chunk逻辑长度≤S，显式尾padding也必须落在S内。
  K个固定槽限制同时工作的group与原BlockingPool job；选下一group最多扫描QG，
  原F2 adapter的resource匹配最多O(T)。总成本包含真实C次IO/原native copy与有界调度扫描，
  不把文件IO、数学prepare/compile或原native等待算作零成本。

## 内存与分配账单

`FileStreamReport::metadata_bytes`只报告共同及copy adapter的已知固定对象/数组，
对应`cold_metadata_bytes`和`FileStreamCopyOwner::report`；它不是整个进程内存总数。
额外成本按原owner明确计费：

| 成本 | 原owner与边界 |
|---|---|
| K个fiber stack及guard页 | 原Mem FiberStackPool；K≤256，冷取得，原默认stack tier及平台页取整；不包含在KS数据stage或metadata_bytes中 |
| File/Mapping控制 | StreamFile/StreamSource冷adopt/borrow使用注入allocator，原File/Mapping资源按其自身合同保留 |
| request取消节点 | 每次成功接纳使用原CancelToken根或派生节点，可能分配且可能返回OOM |
| group及IO coroutine frame | 原Task/IO使用注入allocator；每个实际group/chunk有真实frame生命周期，OOM原样返回，发布前销毁子frame |
| lease控制与F2重放 | 原F1/F2注入allocator与ResourceBudget账单；stage重新获得writer可能涉及原lease控制，不宣称零热分配 |
| native等待及mapping page fault | 每固定slot内嵌一个原BlockingJob，提交到借用的原BlockingPool；没有每chunk新增heap job或私有线程池 |
| driver/vendor内部控制与物理驻留 | `native_metadata_unknown`如实保留UNKNOWN；逻辑uses/allocator归零不能证明驱动私有内存或VRAM归零 |

`FileStreamCopyOwner::submission`只串行化原bind/prepare/submit主机序列，提交后释放锁，
不跨IO或真实native wait持锁。CPU/CUDA/Apple/Vulkan保留各自原生Program/Invocation/Submission，
共同层没有复制或数学kernel、万能Queue、第二runtime、poller或模型缓存。

## 实际路径证据与限制

[Windows往返收据](../../../../evidence/correctness/foundation-f3-math-roundtrip-windows/receipt.json)、
[Air往返收据](../../../../evidence/correctness/foundation-f3-math-air/receipt.json)与
[WSL收据](../../../../evidence/correctness/foundation-f3-wsl-math-exported-drop/receipt.json)
证明实际Qwen范围每路线逻辑payload75,497,984 bytes、stage2,097,152 bytes、峰值两slot，
read/write/reload三程实际native成功复制226,493,952 bytes。consumer逐byte对原始范围，
Direct/Graph输出对独立参考；没有读取其余4.55GB checkpoint或增加整shard暂存。

小样例记录native calls、host waits、stage及actual physical bytes；四路线等待成本沿原native实现，
Apple实际主机等待也计入，不能用“两queue”或异步函数名推断吞吐重叠。
EOF保留已读138/已复制128、GPU后缀失败保留已读320/已复制256与前组ready；
[ASan](../../../../evidence/correctness/foundation-f3-asan/receipt.json)、
[UBSan](../../../../evidence/correctness/foundation-f3-ubsan/receipt.json)和
[TSan](../../../../evidence/correctness/foundation-f3-tsan/receipt.json)各十二项风险实际通过，
真实POSIX short write保留written80/copied96/ready0与准确prefix，跨线程释放前记录不能复用。

正常、OOM、取消、丢弃及native未知观察测试在实际退休后断言注入allocator balanced与active uses归零；
未知时保留owner，恢复观察后才回收。CUDA sticky terminal的物理释放继续沿F2原限制，
不reset用户共享primary context。当前没有需要P1裁决的新性能反例，按活动卡仅做P0。
