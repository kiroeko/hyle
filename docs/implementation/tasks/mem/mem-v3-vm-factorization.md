# Mem v3：VM/storage owner 定向迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem / `mem-v3-vm-factorization` / `REOPEN`；
- baseline：`f077a34c73fba624cbcfaedd2b5cf8dd31655847`；前轮 `mem-v2`；
- trigger：`MEM-VM-FACTORIZATION-2026-09-03`；
- 具体新事实：Platform v3 已接受的因子图要求同步 fs 与 host facts 分开，但当前 Mem public mapping
  在 layout 中直接持有两个公开 Platform storage owner；这既是 Mem public dependency/layout，也是真实
  source/static/shared delivery shape 变化，命中既有 reopen trigger；
- 当前授权仅登记 trigger。完整 capability trace、API/owner freeze、有限 FIX、预算和验证矩阵必须在
  `SCOPED/AUDITED` 后才能修改产品。
