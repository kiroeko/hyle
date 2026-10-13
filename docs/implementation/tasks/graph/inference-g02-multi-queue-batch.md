# G02 多 Queue batch 协调

状态：AUDITED，2026-10-13；owner：`graph`、`graph_batch`。
baseline：`743783f7b0759d77978c0d208dc5ba0898313e69`。
授权来自当前[缺口 goal](../project/inference-gaps-goal-execution-2026-10.md)；
reopen trigger 是新增真实 CPU＋GPU/多 lane 分页消费者，原 F5 coordinator 只保存单 Queue，
且 `BatchGraphAccess::enqueue` 必须匹配原计划 Queue。原 F5/A 的有限 CLOSED 证据仍保留，
不能凭单 Queue submit 改成 span 就假装每个原生 slot 的身份已对齐。

主旅程：host/CPU辅助节点→GPU batch→后续消费者；至少两条实际活跃lane，batch region
计划/metadata/执行在对应原 Queue。正交反证：同设备不同Context/Queue的两个数值段，
防止只按device ID匹配；独立invocations及原结果保活。完整Apple/Vulkan hybrid分页接口
由G02后续native批次完成，本批只签common compiler/leased coordinator，不结案整个G02。

## 最终形状与有限 FIX

冻结公开 `graph::batch::LaneBinding` 保存 lane、Context指针、Queue指针；create接完整
active bindings span、host workspace、原lease options与ReplayOptions。输入只借冷创建调用，coordinator
保存原 QueueBinding 集；Context/Queue/资源沿原借用合同活到native uses与结果退休。
不保旧单Queue签名、不加兼容转发。原普通Graph layout/submit成本不增加；visitor增加lane
身份只用于显式冷/idle桥。其Context/Queue与Graph原targets身份必须一致。

| FIX | 实现与边界 | 验收 |
|---|---|---|
| MQ-01 | RegionAccessView导出原step.lane，保同原region/slot，没有另建plan | Executable/Invocation两种visitor取原lane；不猜ordinal或provider family |
| MQ-02 | 新LaneBinding完整校验并复制QueueBindings；按region lane准备原BatchProgram/native slot，原full bindings prepare_replay/submit | 空/漏/重复/非活跃lane、空/foreign/closed Context或Queue、不同Context同device拒绝；两lane/独立slot/错误/OOM |
| MQ-03 | 迁移全部当前真实batch coordinator调用，native CUDA consumer增加两活跃Queue及plan/metadata实际运行 | 不留旧签名；现有单lane数学/权限用新span；原native队列一致性仍检查 |
| MQ-04 | 更新batch guide并记冷metadata与warm提交结构费用 | public source/static/shared可发现；无默认库新增线程/分派/等待 |

SELF只拥有组合与身份，BORROW全部原数学、F2上传/规划、Graph replay、权限与actual retirement。
不修改batch数学/metadata encoding/页权限，不重写Queue、compiler、allocator、构建/公共工具。
直接依赖与消费者只从中央图和selector生成，不另维护依赖表。

## 验证与停止

结构P0：冷O(lane capacity＋active bindings＋regions)有限metadata，warm沿原Graph queue验证/提交，不逐步复制全bindings
或新plan。保原FI plan/host waits/private UNKNOWN；本批无新吞吐宣称，不启动P1。

预声明：graph与graph_batch适用unit/逐头及selector实际直接消费者；具名native CUDA
coordinator与两Queue数学/权限/生命周期consumer，在支持FI的Linux实际运行，Windows
保其原明确不支持FI域；普通源/installed static/shared对应改动入口。host队列映射/OOM/
失败/关闭用实际consumer与适用ASan/UBSan/TSan；GPU/driver整体风险不伪报宿主san覆盖。
其余provider数学/旧目录不重测；复用对应tree/cache，普通编译从12 jobs按内存调整。

短API/owner审计已固定上述FIX进入AUDITED。原enqueue严格检查slot.queue==实际Queue，
因此按RegionAccessView.lane选择native slot；不是仅提交时换span。ContextRef::owns加原
prepare_replay验证完整target/Queue身份，输入span不被借到热路径。错误/OOM在接纳native
工作前回滚。不同region能用不同Context/Queue，metadata只有原共享资格成立时才共享。
正确性/三交付与适用风险未取得不得CLOSED。
时间为检查点，正常构建/定位自动延长且保原日志/elapsed/FAIL，不以1800秒机械退出。
真实外部阻断按协议裁决；发现改变此API/ownership的事实回AUDITED，不扩大成新runtime。

七问与结果集中留本卡：第三方仍原锁、数学仍原承载、无FFI/新组件图；权限/并发与三交付
按上述consumer验证；复杂度P0、guide与终局源码反证在candidate完成后追加实际记录。
当前产品代码未修改，所有新验证仍未运行，不写PASS。SCOPED提交084dc9a50保存授权顺序。
