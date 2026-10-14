# G03 原 Arena 页的有效初始化前缀

状态：SCOPED，2026-10-14；owner compute_batch；baseline
`cdb8fb9ba251d04f481df02a1e48df5b3a20d890`。这是
[主 pending 消费卡](inference-g03-pending-page-consumption.md)的初始化合同，G01–G10 全范围保留。

真实需求：同一 physical page 在 decode 第一步只写一个 token row，后续追加。
原 PagedPageWriter.publish()/complete() 只能承诺整页 ready，原 PagedArena.read 不区分
有效行和未初始化余量。新 pending frame 必须能判断前一代已初始化域与本次真实写域的并集；
不能靠裸容量、ready bool 或默认填充整个 KV 历史冒充该事实。普通数值分页读取/追加是
正交消费者，机制是连续有效行前缀，不下沉 request、token、evict 或模型策略。

固定公开形状：原 PagedPageWriter 的 publish/complete 增加 optional initialized_rows；
省略承诺整页有效，显式值为 [0,page_size]。initialized_rows() 返回本次 acquire 保留的
已提交有效前缀。原 Arena 存原 identity/generation 和已提交行数；read 按 immutable layout
对每次引用实际需要的行数检查。重复/shared page 取所有引用的最大需求，非尾页需要全页。
声明由实际 producer 成功后提交；不读取/猜测 payload 值来推断初始化。未来 pending 合并
须用原 accepted-copy 覆盖证明扩展域，不由本卡凭调用参数替 pending 签成功。

有限 FIX：IP01 原页有效前缀存储/原 writer acquire/move/publish/complete；IP02 ready frame
拒绝不足初始化域；IP03 64次 acquire/append/old reader/原F2接回，独立byte oracle与明确
OOM/非法行数不消费。使用既有 padding 容纳两个 uint32 字段，目标 ArenaPage48B、
PagedPageWriter40B 在64位机不增加；不加锁/原子/线程/新的controller，原 source/native/数学
实现不修改。完成 audit/固定费用后再产品实现。

预声明 source、relocated static/shared及相关独立头；原 PagedArena9与新增有效域、共享页、
代际、旧reader、显式F2完成、first acquisition/reset、非法数值/OOM；原 ready 数学消费 smoke。
Windows 定向 MSVC，Air 实查仅Mac16,12；三host sanitizer用原同配置树，源码单向复制，
heavy串行/至多4jobs，不启动本地重型GPU或旧全模型。普通docs/closure/selector保留skip。
P0固定行元数据/一次 O(page_refs) 检查，不改数学，不重复P1。调查/实现/验证各2h检查点，
沿用户常设授权追加，保原FAIL/累计；有限本卡通过后停止，不代签pending/source/native late
output/真实Graph/七路线64步。主卡仍待完整receiver形状，不能重开已闭数学/工具。

七问：无需新依赖/FFI；推理追加与普通数值页两旅程；借原F1/F2和原Arena，无第二allocation
系统；P0增加常数行状态且复用padding、一次引用域扫描；三种C++交付签名/导出一致；guide
说明有效域声明和旧reader；原未决能力全部保留。风险门未运行不写PASS。
