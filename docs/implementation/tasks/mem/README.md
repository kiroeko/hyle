# Mem 任务索引

当前产品状态只见 [`module-status.json`](../../module-status.json)。本目录保存 Mem 各轮施工与回放事实，
历史 task 中的“下一步”、旧 API 和性能结论都不是当前授权。

## 当前回放

- [`mem-current-product-replay-2026-09.md`](mem-current-product-replay-2026-09.md)：对当前
  `hyle::mem` 产品及其 `mem/mem_vm/mem_reclaim` 因子执行 product-non-mutating replay。先回答完整
  功能超集、自然 C++ 形状、第三方与直接依赖、构建交付、当前性能证据和文档是否仍成立；只有产生
  `PRODUCT_TRIGGER` 才建立新的产品 closure round。

## 当前产品身份的累积来源

- [`mem-v2-rebuild.md`](mem-v2-rebuild.md)：Mem 产品边界、资源/Buffer/池/规划与 host allocator 重构；
- [`mem-v3-vm-factorization.md`](mem-v3-vm-factorization.md)：`mem_vm` 因子化；
- [`mem-reclaim-v2-inference-resume.md`](mem-reclaim-v2-inference-resume.md)：`mem_reclaim` 恢复轮；
- [`mem-v4-reclamation-facade.md`](mem-v4-reclamation-facade.md)：reclamation 进入 Mem 产品 façade。

其余文件保留更早的实现、缺口调查和失败经验，必须结合上述当前身份与 registry 阅读。
