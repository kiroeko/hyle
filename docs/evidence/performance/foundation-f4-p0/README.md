# F4 candidate P0 成本结构与 LOC

固定29d3c7a3e；本记录只裁决结构成本，不给timed WIN/PARITY，也不代签SDK和终局红队。
源码hash、MSVC shipping Release对象hash及publish生成代码摘录在receipt与zip中。

E为schema成员数，V/D为版本/draft容量，B为retained block容量，C为clone slots。
目录冷分配O((V+D)E+B+C)，共享version/identity只增加store/record原子引用，O(1)，不复制
payload。begin/finalize与末个record释放O(E)，找空slot另有O(V)或O(D)扫描；不声称分支
全过程O(1)。metadata snapshot有确切byte copy与原allocator/资源成本。

ledger去重是有限冷扫描：stats有O(B²+BC)，import preflight含O(EB+E²)及stats开销，
单成员replace也有上述有限ledger核对。默认B=128，clone slots=4；没有全局hash表/逐Tensor
字段，也不能称任意大page目录已具备F5热路径资格。保留的reader/native use继续计费，
同原F1 pool按全backing容量一次计费；不因目录句柄释放伪造payload已回收。

成功head发布的临界区是store局部monitor、身份比较与handle交换，O(1)，不分配payload、
不等GPU/IO。旧record释放在临界区之外，其末引用仍可能O(E)并重新短暂锁store。生成代码
0x1c0先unlock，0x1da再drop_record；其他临时identity只有非末引用。失败Status消息沿原
Error机制可能分配，不能把全部错误路径签成无分配。调用本身不是lock-free承诺。

dirty clone只对一个完整byte-addressable成员走原F2，成本O(dirty bytes)，原native calls、
host waits和未知metadata按CopyBatchReport保留。prepare/rearm是冷路径，wait/drain显式；
取消不保证GPU抢占，opaque raw clone明确拒绝。新对象的slot observation mutex只保护显式
host observer；普通Tensor/Graph Invocation无新增常态state字段或每次执行的版本扫描。

主/正交消费者实际只克隆conv window和所选KV块，pure recurrent/native final readonly
直接replace，旧版本数值保持。initialize、oracle下载、staging更新均有成本，不据此声称
单机推理吞吐已达标；F5分页数学与大规模batch planning仍未实现。

统一loc-report.py、cloc2.08在同一不可变Git tree统计，self-check PASS：全库2,505,378，
自实现不含测试171,728，新compute_state完整实现边界1,147行/5文件（显式包含3个公头）。
仅按src/compute_state的687行会遗漏公头，不能当完整模块数；也不把测试和依赖行数算进
引擎需要自行实现的产品代码。完整口径、路径与计数在loc.json。
