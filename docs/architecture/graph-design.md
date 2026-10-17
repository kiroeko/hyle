# Graph：逻辑计算与原生执行计划

Graph把模型无关的typed DAG编译为可重复执行的程序，拥有边界、依赖、放置和资源规划。
Compute拥有直接计算与原生执行，Provider拥有片段内部程序；compiled plan不是第三个公开产品。
实际用法见[用户Graph指南](../user/graph.md)，共同实现选择见[Compute冷路由](compute-design.md#冷路由)。

## 图、制品与运行状态

逻辑图描述多输入/输出、常量、operation/custom execution、View/Copy、alias、资源访问与控制顺序。
数值与物理编码是合同，不能将opaque/quantized bytes当普通标量，也不用vendor op枚举作公共语义。

不可变制品共享计划、代码、常量和合法packing；并发运行槽独占bindings、scratch、mutable native
state和结果。同一槽未完成不能重绑、覆盖或重放，排他状态持有到终态，不只保护enqueue。
共同Invocation接纳后失败会退休该槽，后续使用须重建；接纳前拒绝不等于槽已失效。
输出可由caller、Graph或Provider拥有，来源alias也保留真实owner；不为统一ABI制造多余复制。

编译不偷执行用户输入或改变副作用。常量borrow/materialize/packing具有明确owner；兼容性未知时，
不能因地址或大小相同就共享。跨制品cache由消费者显式拥有，不建立隐式进程全局cache。

## 形状与副作用

图可表达有限exact形状与有界symbols/guards。flatten、concat、head split、broadcast等派生关系
由只读shape program维护，避免引擎重复维护中间相等关系。checked加乘、整除、广播与typed guards
足以覆盖这些需求，不引入通用符号求解器或模型脚本解释器。

接纳前核对overflow、元素数、stride、alias span、workspace及整个域。边界固定不能掩盖内部动态；
无法证明全域时缩窄域或选择exact形式，不以端点抽样代替证明。View默认保留实际stride/alias，
不能因没有kernel而自动丢弃资源、控制边或强制dense复制。

schema声明资源的read/write/read-write。调用者用data/control边表达同一canonical owner/subrange冲突的顺序，
图构建结束时验证；node添加顺序不是effect依赖，Graph不自动补边或用全局effect锁串行化无关工作。
可证明独立的资源可并行，未知副作用不能谎称pure。
跨Invocation共享可变状态由调用方约束。Provider须保存片段完整副作用，不能只删拒绝条件。

## 冷编译与有界规划

编译先验证语义/描述，匹配随实现交付的规则，再确定合法region和边界、原生顺序、workspace与制品。
region是可收缩的node set，不必连续编号；收缩不能产生离开后重入的新环，选中片段不重叠并完整覆盖执行节点。
只处理有限提案，不枚举任意DAG的所有子集。

硬约束先于性能选择。Graph根据完整计划的copy/repack、同步、dispatch、融合与workspace作有界结构判断，
不在模型准备时试跑多个Provider争夺赢家，也不要求消费者提供timing profile。
候选覆盖节点更多或局部kernel更快，不代表完整计划获益。

有界规划不能找到可行计划时明确拒绝；已找到合法计划但优化预算耗尽时可报告优化受限。
这些预算限制计划搜索的计算成本。只有接纳前、无副作用的能力/资源拒绝可尝试后备，
真实错误不吞掉，接纳后不换实现重算。

## 并发、内存与时间线

先根据data/control、lane顺序及跨lane依赖建立happens-before，再决定backing复用。
不能仅按拓扑编号区间忽略并发。默认保留并发，只有明确允许才增加可报告的串行依赖换内存。

Graph给自身backing与必要转换调用Mem的lifetime/quota机制；Provider私有heap和生成输出由原owner规划。
报告区分编译峰值、唯一常驻权重/制品、有限variants、并发scratch/capture、外部借用和未知driver/private成本。
同一allocation不重复计费，未知不当零；仍有capture或异步引用就不回收。

target表示可执行条件，storage表示数据域，lane表示顺序；多stream不制造多个假设备，
unified memory也不取消依赖。第三方时间线保留原生对象，不要求伪造Hyle Queue。

## 原生组合与重复执行

单路线及CPU＋GPU组合使用同一图语义，保留不同compiler、输入绑定和完成方式，见
[路线设计](compute-routes.md#cpu与gpu组合)。不以mixed SDK编译成功推导全部混合数学或硬件支持。

稳定prepared replay与声明有metadata/array等成本的managed执行分别表达；managed也不允许隐藏JIT/search/repack。
批次native planning属于提交前独立准备，capture能力与是否阻塞另行表达。形状或相关数值/资源合同改变，
须重新准备或使用预先有限的variant，不能在submit里重搜Provider或无限增长bucket。
分页布局准备和状态发布见[Compute设计](compute-design.md#分页计算为何需要独立准备尺度)。

参数化模型/资产版本变化时的通用增量失效层仍是未来需求，独立于当前typed DAG、有限variants和引擎KV策略。
若需要，应另外说明版本identity、依赖与重算边界，不给每次submit预埋dirty字段或提前制造新API。

取消停止后继，已接纳工作保活至真实终态。结束使用时先停止新调用，再显式等待/退休本owner已接纳工作；
不能假定名称叫close就会阻塞drain，也不把借用的Context、executor或allocator当作可销毁的私有资源。
