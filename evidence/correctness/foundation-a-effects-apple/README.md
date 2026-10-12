# Foundation A4 Apple 公开 source consumer

canonical 实施 commit f36bf4c1de6aa1f5a3d9383fd2e4e8ac9ab9e7ad；完整 archive/checksum 及
实际 Air 身份见 source-sync.json/job.json。Mac16,12 /16GiB/macOS27.0.1/Clang21；未用 Mini。
只 build 新 public effect target，复用 shipping CPU+Apple source 树，按实际内存 j1/test1。

普通 [公开头 C++ consumer](../../../tests/graph/public_effect_consumer.cc) 的本快照版本通过
CPU resource write→Apple GPU math→CPU resource read、cross-fragment alias 拒绝、accepted
failure 保留实际写入但不执行后继、zero-output effect，以及 close 后 old result/用户 Context
再次使用。没有测试专用产品接口或新数学/执行器。

本 receipt 对应 f36 快照，尚未包含后来红队的 cancellation gate/must-alias 修复；
最终候选会定向核验这些适用项。installed、独立头/缺路线/迁址、sanitizer/最终红队未由此代签。
