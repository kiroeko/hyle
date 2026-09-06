# Container Text v1：文本、路径与标识因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round：`text` / Container / `text-v1-factor`；
- baseline：`b5568e929787ecd5bb222e436c7456c6517fae21`；触发与产品总合同见
  [Container v5](container-v5-product-factorization.md)；
- scope：接管 StringView、ASCII/string algorithms、lexical path、StringInterner/StringId 与 compiled
  string/UTF kernels；补齐审计准入的 typed Unicode error offset 和 UTF-8/16/32 caller-buffer conversion；
  建立独立 target/façade、provider visibility、真实 loader/tokenizer P1 journey；
- non-goals：owning string/Cord/IOBuf、filesystem access、locale/normalization/grapheme、tokenizer/trie/
  radix/model automaton、protocol、Native FFI/C API、稳定 C++ ABI、公共仪器和 Formal；
- candidate dependencies：PUBLIC `core,mem,container`，PRIVATE `platform`；simdutf/Highway 只能是
  private/build-only mechanism，不公开 provider type、symbol 或第二套 ISA authority；
- tier / machine / margin：P1，`closure-perf-v1`，W1-WIN，5%；W1-WSL/M1-MAC correctness、真实 ISA
  与交付；
- budget：审计 45 分钟、实现与测试 2 小时、性能 45 分钟、终态红队并入 Container suite；最多一次
  performance-guided product correction；
- next：完成 public trace、Unicode result/output shape、provider borrow/visibility、普通调用草图和有限
  FIX ledger 后进入 `AUDITED`；此前不修改产品。
