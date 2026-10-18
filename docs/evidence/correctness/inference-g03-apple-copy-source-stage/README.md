# G03 Apple 固定 F2 原生只读来源 source 证据

候选 dab23ba92，2026-10-14，MacBook Air Mac16,12 / arm64 / macOS27.0.1 / Apple Clang21。
[源码身份](source-identity.json)逐字节核 canonical 与验证副本；[累计记录](result.json)、
[job](job.json)及[全部输入/日志/编排](raw.zip)。这一阶段全部 actual terminal 0。

- [公共数值来源](apple-joint-2.log)：64 次真实原 Abs producer，固定一个 F2 program/两
  slots；保前一步 readonly export 完整 bits、部分来源/目的 range、超容量与 host/普通
  异资源/相同 domain 伪造拒绝，原 mutable rejection 不消耗 writer；注入 allocator
  的 bind/prepare 调用计数不增加。readonly import 冷费用与显式 ready 等待仍保留。
- [完整 ready 数值旅程](apple-native-producer-page-1.log)：F16/BF16 各64步、固定原
  producer program/slot 与 F2 program/slot，native Abs K/V 输出不下载host结果，精确
  dirty196608bytes/组、原页同generation/未改历史、两request/page16跨页、GQA8/2D256
  Q1/17、原 attention 独立 double oracle、frame权限/未引用页/旧result保护。
- [原 copy 单测](apple-copy-unit-1.log)和[实际 copy 风险](apple-copy-risks-1.log)：
  原权限、original native producer、phase/prefix、OOM、取消/放弃/关闭与失败回归。
- [直接联合图](apple-graph-direct-1.log)：原 CPU→Apple paged→CPU 公共合同回归。

源 target 都只选真实直接消费者；公共 probes 仅 hyle::compute，联合图仅 hyle::graph，
私有 copy 单测不冒充 SDK。所有 build 最多4 jobs，heavy 串行，未跑全模型或旧数学 P1。
旧形状反证的四个 fixture输入/FAIL 仍在独立原证据，未清除失败或重置预算。
这签固定 F2 接纳 ready readonly exports 的有限来源修正，不签 pending page/attention、
零整步分配/零host waits、整个 G03 或十项 goal；宿主san与两SDK仍待后续实际验收。
