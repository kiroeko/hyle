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

## Audit verdict（`AUDITED`）

- public trace：`StringView`、string algorithms、path、StringInterner/StringId 与 compiled string/UTF
  sources 归本 factor；semantic hash/PodVector/FlatHashMap 归 `container`，filesystem access 归
  FROZEN `platform_fs`；
- borrowed text/path/interner/ASCII：`COMPLETE/KEEP`；StringView 的 4 GiB limit、borrowed lifetime、
  known-NUL bit和 embedded-NUL 陷阱继续明确；owning text 直接用 standard/PMR；
- Unicode codec：`PARTIAL/FIX`。新增 `UtfStatus` 与 `UtfResult{status,input_offset,output_count}`，支持
  validate 及 UTF-8/16/32 各方向 caller-owned span conversion；malformed surrogate/codepoint、output
  too small、empty、exact-size query 与 non-overlap/alias contract 显式；
- mechanism：simdutf 只负责经版本锁定的 validate/transcode core，HYLE 拥有 public enum/result/span
  contract；Unix hidden visibility，public header 不出现 provider type/macro；
- dependencies：PUBLIC `core,mem,container`，PRIVATE `platform`；不依赖 `platform_fs/simd/sched/io`；
- P0：每个 codec O(input+output)、单 pass/至多一次资格化 size pass、无 allocation。P1 primary 为 model
  metadata UTF-8→UTF-16/32 transcode，SAME_CONTRACT 对齐 validation、result、allocation 和 output-size；5%；
- finite FIX：`TEXT-F01`（独立 target/codec/tests/direct edges）、`TEXT-F02`（provider visibility/delivery）
  和 `CONT5-F06`。normalization、grapheme、regex、tokenizer/matcher 不进入本轮。

## Candidate（`CANDIDATE`）

- product commit / tree：`31c3ace558a821902d689d2d49c7bf8a102d5beb` /
  `b10d2261aecbcac34ad8d77a7fa8adc25e78bf9c`；
- `hyle_text` 独立拥有 borrowed text/path/interner 与 compiled codecs，PUBLIC closure 为
  `core,mem,container`、PRIVATE `platform`；六方向 caller-buffer conversion 及 typed malformed offset/
  required count 不公开 simdutf 类型；
- W1-WIN shared/static header、empty、malformed UTF-8/16/32、output-too-small、六方向 conversion、tail 与
  真实 ISA tests 通过；Unix private-symbol scan 和 simdutf SAME_CONTRACT P1 等待 clean candidate 验证。
