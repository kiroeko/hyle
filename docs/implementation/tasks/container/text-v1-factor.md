# Container Text v1：文本、路径与标识因子

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

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

## 第二 Candidate

首候选的 exact-contract UTF P1 显示全分层固定约 3.1–4.1% 差距，部分置信区间跨过 5%
non-inferiority boundary。本 factor 在不改 API/语义/layout/workload 的前提下，让单次
safe Unicode 操作的 validate/size/convert 复用同一 operation-local simdutf implementation
pointer。新 product commit/tree 为 `a7ce0102413c783e26227c68645e0282b109fde4` /
`6ae52a087b8b2b9271bbd98acb9d7c62bb61d10c`，全部门必须重放。

## 第二候选撤回

`hyle_text` 自身的 simdutf hidden-symbol gate 已通过；但共同 `hyle_container` exact shared artifact
泄漏 24 个 private bitset ISA symbols，因此共同产品身份仍须退回 `AUDITED`。本 factor 的 API、layout、
算法与 operation-local provider 修正不变。第二候选 UTF 八个分层 median 为 `0.9616–0.9731`，其中
三条置信区间跨过 5% 边界，工程 verdict 保持 `INCONCLUSIVE`。

## 第三 Candidate

共同 product commit/tree 为 `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` /
`8025ee111380c2600f804cf979ed59d289a8be76`。本 factor 产品没有再次变化；W1-WSL GCC 14
shared/WERROR tests/header tests 2/2 通过，shared dynamic symbol scan 保持 simdutf/provider 0，
公共 Text symbols 39。

## Terminal verification and decision

精确共同候选 `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` 已通过 Windows static/shared、WSL
ASan/UBSan/WERROR、macOS UBSan/WERROR，以及三平台 source/static/shared relocated consumer。六向
UTF-8/16/32 conversion、malformed offset、size query、output-too-small、borrowed lifetime、private
provider 与动态符号门均通过；WSL/macOS 的 simdutf/provider 导出为 0、公开 Text symbols 为 39。终局
红队未发现新的产品 defect、owner 泄漏或 benchmark-shaped API。

本 factor 从 `CANDIDATE` 终止为 **`BLOCKED`**：scope、architecture、correctness、delivery、
usability/docs、red-team 均 `PASS`；engineering performance=`INCONCLUSIVE`。第二候选有效输入自然路线
HYLE/peer 为 `0.9616–0.9731`，三条 CI 穿越 5% 边界；更关键的是 HYLE 返回 typed status/error offset/
output count，而 peer adapter 对错误输入终止，结果合同未等价，不能签 `SAME_CONTRACT` non-inferiority。

- blocker owner：`container-v5-p1-contract-evidence`；
- resume condition：新 `RESUME` round 等价化 validation、typed error/result、output-size 与 caller-buffer
  contract，在 immutable candidate、W1-WIN、5% margin 下取得合格 verdict；只重测 valid-input
  natural route 不满足同合同恢复条件；
- known limitations：Text 不拥有 locale、normalization、grapheme、regex、tokenizer/model automaton 或
  filesystem I/O；caller buffer 不允许未声明 overlap；当前没有同合同 P1、全面 SOTA 或 Formal 结论；
- reopen triggers：真实 consumer 暴露 correctness/security/lifetime/API/能力缺口；Unicode/path/string
  contract、provider/ISA、依赖或三交付改变；强 peer 证明 material regression 或更优复杂度档。

LOC 绑定共同 immutable candidate：全库 464,718；自实现包含测试 121,614；自实现不包含测试 77,078；
`text` 实现 2,040。

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->
