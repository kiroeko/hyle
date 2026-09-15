# R2 共同语义与 Mem 定向尾项验收

产品候选：`4e9f74a09b7218f6053038cc5c5b091e73ff0bc5`。
只修正测试可移植性的 revision：`e25b27f0bf9ec5a6d0b31ec6b49c7bdbf69ea256`。
两者之间的 include/src/CMake 产品差异为空；Windows 为产品真相，Unix 只有出站验证副本。

## 实际完成

- D21–D24：独立 Direct delegated prepare/submit、冷依赖存储、真实 Completion 复合保活、typed
  OwnedResult/ResultLease 和可复用 ResultSlot；真实 CPU 数值及 CUDA kernel 终态/放弃后回收。
- G31–G35：删除用户 timing profile 与多计划计时竞赛；首个满足既定规则和资源硬约束的完整计划
  即绑定。并发要求直接表达；共同 Compute-backed 编译只接同一 Device，保留同设备多 Context、
  Provider、lane、host storage、packing/常量共享、预算与事务化回滚。
- Mem M100/M101：补完现有 Reservation 的平台、sanitizer 和 Graph 消费尾项；没有修改 Mem 产品。
- R2-Q schema 证据仍由独立已关闭的 ComputeOps 卡持有，本目录不重复签其 kernel 能力。

## 验证结果

| 范围 | 实际结果 | 原始证据 |
|---|---|---|
| Windows 不可变候选 selector | 14/14 PASS，82 项保留跳过/复用理由 | windows-raw.zip 中 selector-result 与 receipts；selector-summary.json |
| Windows 同一 owning-result consumer | source/shared/static 各 1/1 PASS；使用已 relocated SDK | windows-raw.zip 中 owning 日志及 owned-consumer 源码 |
| Windows 测试写法纠正 | 5/5 CTest PASS，仅 Compute/CPU/Native/Graph 受影响门 | windows-raw.zip 中 r2-test-windows 日志 |
| WSL GCC 14.2 | Mem/Compute/CPU/Native/Graph 与逐头 11/11；拥有式结果 source consumer 1/1 PASS | linux-logs 中 gcc-portability-final-tests.log 与 off/owning-source-tests.log |
| WSL ASan / UBSan / TSan | 各 6/6 PASS；TSan 按项目合同使用 setarch -R | linux-logs 中各 sanitizer/tests.log |
| MacBook Air | Apple Clang 21；共同代码 11/11，source consumer 1/1 PASS | air-logs |
| Mac mini | Apple Clang 21；共同代码 11/11，source consumer 1/1 PASS | mini-logs |

各构建为 Release/WERROR、实际并行 2；重型验证按 Windows → WSL → Air → mini 串行。
Mac 明确使用 macOS 14.4 deployment target。Unix 共同语义门使用 CPU/Native，CUDA/oneDNN OFF；
Windows 的 full selector 另外覆盖真实 CUDA、oneDNN 和 Graph-oneDNN。这里的 Unix 三机消费是
源码嵌入及 producer DSO，不伪称每台都重做了 installed static/shared SDK；Windows 三交付已验证
共同 C++ 接口，R3–R6 各自仍须完成 native 路线的适用三交付与完整默认入口。

Windows 命令为 validation-select.py run --base fb2fc4e5 --candidate 4e9f74a0，context/完整命令均在
windows-raw.zip。测试纠正的 selector plan 在 r2-test-portability-plan.json；其选定 configure 与
对应 factor 门在既有 Windows Ninja 缓存回放，没有重建 SDK、没有 P1。Unix 完整 recipe 在
unix-recipes.zip；跨 DSO consumer 与原脚本、修正脚本一并保留。

## 首轮失败及纠正

- 实现期 MSVC 对齐形参、CUDA 首次调用 gate 和后台实际回收时点、Graph 旧 fixture 身份/断言迁移，
  保留 D/G 原始失败与最终回放日志；没有将初始失败覆盖为 PASS。
- GCC 首轮在既有测试的宏断言括号、range-loop 复制、自赋值写法和 aggregate 缺省初始化处报 WERROR。
  8 个测试文件定向纠正，断言内容保留；原 build、集中诊断与最终 build/test 日志均在 linux-logs。
  Windows 仅回放受影响测试。r2-test-portability-files.json 固定这 8 个文件的 SHA-256，三个 Unix
  副本的 test-overlay-sha256.log 均核对一致。产品源码和算法没有候选纠正。
- Air 产品和测试程序编译成功后，外部 recipe 的空数组触发系统 Bash 3 的 nounset。修正 recipe，
  从未运行的测试继续；原脚本/失败 wrapper 与完成日志都保留。没有改产品或关闭 WERROR。

出站基础 archive 的 SHA-256：
`9bfab18d206391024e411b087444c5f5bdabfe7712f8abd2d7895c159fafd8dd`。
验证目录仍沿基础候选名 source-4e9f74a0；其测试仅应用上述已提交 e25b27f0 的出站修正。

## P0、限制与结论

源码红队见 r2-red-team-4e9f74a0.md。ResultSlot 一次冷分配 metadata 和对齐 T，T 自己拥有 native
payload；新 Direct 独立状态一次冷分配依赖 scratch。新可选热路径有明确 admission/refcount/有限
依赖转换，不宣称零开销。Completion 和既有 backend event slot 布局不增字段，释放先清附件节点
再归还 owner；Graph replay 不新增测量、搜索或分配。Mem 原 allocator 路径/布局不变。

borrowed allocator、executor、输入绑定与依赖 Completion 仍遵守各自保活合同。ResultLease 保留
原 native owner；同槽被执行或旧 lease 占用时拒绝覆盖。真实失败、取消与放弃不能提前回收。
协议 fixtures 只证明边界；本目录不签正式 MLX/Vulkan、CUDA W4A16、CPU 默认优选或路线 P1。

P0 工程门 PASS。本包没有运行 P1，不声明 WIN/PARITY，也不以本轮公共语义替代 R3–R6 的同实现
包装税、完整片段性能、默认规则与资源验收。R2 和 Mem 本轮七门通过，可关闭本范围。

统一 loc-report.py --self-check：e25b27f0 的 Compute+Graph 实现 **30,958** 行，自实现不含测试
**126,143** 行，全库代码 **692,165** 行；测试修正没有增加实现行数。候选与最终测试 revision 的
原始统计均在本目录。文件校验见 sha256.json。
