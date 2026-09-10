# Container Text v3：Base64 / Base64URL 批量编解码

## Closure card（`SCOPED`）

- component / owner / round：`container_text` / Container /
  `container-text-v3-base64`；kind=`REOPEN`；
- baseline commit：`f3f20528ab3deea1dcf3bed5389226dcee75d1c4`；前轮 terminal record 为
  [`container-v6-identity-p1-resume`](container-v6-identity-p1-resume.md)；
- trigger 与证据：已接受的基础能力扩展决策
  [`G01`](../../../architecture/foundation-capability-expansion.md#g01base64base64url) 证明四个推理基座的
  协议/多模态路径需要 Base64，而 `container_text` 已编入 simdutf v9.1.0 却没有可发现的 HYLE 公共入口；
- 正常 consumer 任务：协议、模型元数据或多模态输入在 caller-owned contiguous buffer 间执行标准
  Base64/Base64URL 的 bulk encode/decode；调用方先查询容量，再复用 Container/Mem storage，不为短样例
  隐藏分配；
- 主产品纵切 / 具名来源：llama.cpp/ggml、vLLM、SGLang、ds4 的模型/请求二进制载荷文本化路线；成熟机制
  采用当前 vendored simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`；
- 正交反偏置纵切：低延迟网络协议和未来 WebSocket handshake 使用短、中等 token；它防止 API 只适合
  大型模型 blob，也不授权把 WebSocket 或 owning service façade 塞入本轮；
- 跨模块 public journey 与每段 owner：`container_text` 拥有 alphabet/padding/whitespace、size、typed
  progress/error 与 provider adapter；Container/Mem 拥有 caller buffer；后续 `io_websocket` 只消费该公开
  路线，不复制 codec；
- scope：exact encoded size、checked/max decoded size、caller-owned input/output span；standard/URL
  alphabet；required/allowed/forbidden padding；accept/reject ASCII whitespace；fast decode 的 status/input
  consumed/output written；独立 cold detailed validation 的 first-invalid offset；不足容量 typed failure；
- non-goals：streaming codec、MIME line wrapping、自动 mixed alphabet、owning string convenience、
  WebSocket/TLS/JSON、C 或 Native FFI、稳定 C++ ABI、另一套标量/SIMD codec、公共性能仪器修改；
- 直接依赖 / 直接消费者：依赖保持 `PUBLIC core,mem,container / PRIVATE platform`，simdutf 仍是
  `container_text` 的 vendored package dependency；直接消费者保持 Container façade、Util/IO，未来
  `io_websocket` 经公开头消费；本轮不新增 HYLE component edge；
- 性能 tier 与原因：`P1`。Base64 是可观 bulk 热点；目标不是赢一个人为小循环，而是证明 HYLE adapter
  在同 alphabet、padding、whitespace、buffer ownership、validation 与 result 合同下没有 material wrapper
  tax，并保留 simdutf 的线性 SIMD core；
- 适用机器 / 仪器 / material margin：W1-WIN、Release/static source consumer、`closure-perf-v1`、5%；
  W1-WSL 与 M1-MAC 用于 correctness/portability/delivery，不把同物理机或未绑定 Mac 冒充独立 Formal
  性能证据；
- 调查与纠正预算：一次 API/provider 审计、一个产品批次、一次 correctness/red-team correction、一个
  immutable candidate 和一次正式 P1 qualification（只允许合同自带的精度升级）；不扩建公共仪器；
- 预声明验证：header self-containment；RFC 4648 vectors、empty/short/tail/all byte values、alphabet 交叉
  拒绝、三档 padding、whitespace 两档、invalid character/extra bits/remainder、null/empty、overlap 与
  output-too-small；fast/detailed progress；direct simdutf semantic oracle；W1-WIN/W1-WSL/M1-MAC 定向
  correctness；Container source/static/shared installed journey；API/符号/依赖检查；P0 code/cost 审计；
  W1-WIN P1；终局独立红队；
- stop / blocker 条件：若 simdutf 无法承载显式 whitespace/padding 合同且需要复制 codec、三平台错误语义
  无法统一，或唯一 P1 在一次合法纠正后仍 material fail/inconclusive，则诚实裁决 `BLOCKED`；不得降低
  默认严格语义、改变 workload 或增加第二 provider 来伪造完成。

## SCOPED 不变量

1. 默认必须是 standard alphabet、canonical padding required、ASCII whitespace rejected；URL、unpadded
   或 forgiving 都需调用方显式选择，不能用含糊的 `strict` bool。
2. encode/decode 不分配；公开类型不暴露 simdutf enum、result、implementation pointer 或 option bitmask。
3. 快速失败允许 output 含未提交 partial bytes，调用方必须整体丢弃；first-invalid offset 属于独立冷诊断
   路线，不能强迫正常成功路径多跑一遍。
4. size helper 对溢出、非法输入和策略不匹配必须可判别；不得用 `SIZE_MAX` 同时表达多个公开失败语义。
5. P1 只比较固定 API/correctness 后的真实调用；同 alphabet/strictness/buffer/result 之前不产生胜负结论。

## 进入 `AUDITED` 前的有限审计队列

- 逐项映射 simdutf `base64_options`、last-chunk、safe/full result 与 HYLE 的 alphabet/padding/whitespace
  正交语义，特别确认 simdutf 默认忽略 whitespace、padding allowed 与 HYLE 默认 required 的差异；
- 冻结最小自然 C++ API：enum/options、size result、encode result、fast decode result、cold detailed result，
  以及 `StringView`/`std::span<std::byte>` 的 ownership、null、overlap 和 partial-output 合同；
- 证明 exact/max decoded sizing 是否需要扫描、如何报告 invalid，并避免正常 decode 被迫双遍历；
- 扫描四个推理基座、Folly/simdutf 与未来 IO consumer，确认没有遗漏本轮必须的 contiguous bulk shape；
- 冻结有限 `KEEP/FIX/BORROW` ledger、测试矩阵、P0 成本和唯一 P1 workload。完成这些之前不得修改产品代码。

## 能力与形状审计（`AUDITED`）

| 能力族 | 层级 | 准入旅程/来源 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|
| standard / URL bulk encode | 原子机制 | embedding、文件响应、WebSocket accept；simdutf/Folly | simdutf 已编译但无 HYLE API | `MISSING` | `BORROW + FIX` |
| strict/forgiving bulk decode | 原子机制 | multimodal data URI、环境/协议载荷；四个推理基座 | llama.cpp/ds4 自写 owning scalar，HYLE 无路线 | `MISSING` | `BORROW + FIX` |
| exact encoded / max decoded size | 原子机制 | caller-owned Container/Mem allocation | 无可发现路线 | `MISSING` | `FIX` O(1) checked arithmetic |
| checked exact decoded size / diagnosis | 复合功能 | 配置校验、错误响应和 tight allocation | simdutf size helper 不验证完整语义 | `MISSING` | `FIX` 冷标量 validator；不复制 codec |
| owning string/vector convenience | convenience | llama.cpp/Folly 高层包装 | `std::string`/Container/Mem 可自然组合 | `ROUTED` | 不隐藏 allocation |
| streaming/MIME/mixed alphabet | 协议策略 | MIME、任意 chunk stream | 没有本轮具名 journey | `OUT_OF_SCOPE` | `DEFER/REJECT` |

四个推理基座的共同最低需求是 contiguous standard Base64；URL alphabet 是协议基础能力，Folly 与 simdutf
均直接提供。llama.cpp 的 auto alphabet、in-place owning decode 与 ds4 的函数内 allocation 不是 HYLE 的
默认形状：前者会放宽协议，后者把 allocator/OOM/lifetime 藏进 codec。HYLE 用显式 alphabet 和 caller buffer
覆盖其原子需求；data URI 拆分、媒体类型、JSON 和请求大小上限继续由 IO/service owner 负责。

### API / usability freeze

新公开头为 `<hyle/container/base64.h>`，namespace 保持 `hyle::container`：

```cpp
enum class Base64Alphabet : std::uint8_t { kStandard, kUrl };
enum class Base64EncodePadding : std::uint8_t { kInclude, kOmit };
enum class Base64DecodePadding : std::uint8_t {
  kRequired, kAllowed, kForbidden
};
enum class Base64Whitespace : std::uint8_t { kReject, kAllowAscii };

struct Base64EncodeOptions;
struct Base64DecodeOptions;
enum class Base64Status : std::uint8_t;
struct Base64SizeResult;  // status, input_count/error offset, output_size
struct Base64Result;      // status, input_count, output_count actually written

constexpr Base64SizeResult base64_encoded_size(
    std::size_t, Base64EncodeOptions = {}) noexcept;
constexpr std::size_t base64_max_decoded_size(std::size_t) noexcept;
Base64SizeResult base64_decoded_size(
    std::span<const char>, Base64DecodeOptions = {}) noexcept;
Base64Result base64_encode(std::span<const std::byte>, std::span<char>,
                           Base64EncodeOptions = {}) noexcept;
Base64Result base64_decode(std::span<const char>, std::span<std::byte>,
                           Base64DecodeOptions = {}) noexcept;
```

- options 默认值严格固定为 standard + include/required padding + reject ASCII whitespace；decode 的三种
  padding 都要求 canonical unused bits，`kAllowed` 只表示 padded/unpadded 两种 canonical 拼法均可；
- ASCII whitespace 精确指 space、tab、LF、CR、form-feed；`kAllowAscii` 忽略这些字符，不接受其他垃圾；
- `base64_decoded_size` 是 O(n) 的 checked exact-size 与 first-invalid cold diagnosis；想单遍 decode 的调用方
  先按 O(1) `base64_max_decoded_size` 分配，成功后缩到 `output_count`；
- fast decode 的 provider 错误可比 cold diagnosis 粗，但始终给 input progress 和实际 output bytes。失败输出
  未提交且必须整体丢弃；policy precheck 在写入前失败时 `output_count==0`；
- byte-span 是核心 ABI；`span<const char>` encode 和 `span<char>` decode 只作为 header-inline、
  `std::as_bytes/as_writable_bytes` 便利重载，不拥有、复制或分配。红队编译证明同时提供 `StringView`
  重载会让 `std::string/vector<char>` 在两个隐式 range conversion 间二义，故不准入；现有 view 使用
  `view.as_std()`；
- 输入输出不得重叠；empty span 合法；非法 enum、容量、size overflow 和无法满足的 span 参数返回 typed
  status。调用方不能把 output buffer 当 NUL-terminated string，结果长度只认 `output_count`；
- exported 核心保持 `noexcept`。本地 hot-path 状态不升级为 `error::Status/Result`，因此不为 `container_text`
  增加 Error 边；需要请求上下文时由 IO/service 在边界映射。

### 复杂度与 provider 映射

- encoded/max size 为 O(1)；checked exact size、encode、decode 均为 O(n)，无 allocation、lock、atomic、
  syscall 或 TLS 状态；公开对象只是值类型 options/results；
- encode 路由 simdutf `binary_to_base64`；decode 路由 capacity-safe `base64_to_binary_safe`，每次操作至多一次
  provider dispatch。HYLE 不复制标量/SIMD codec，也不暴露 provider enum/bitmask；
- simdutf 默认忽略 whitespace，且 loose/strict 不能单独表达三档 padding。HYLE policy precheck 只补这些
  语义：默认拒绝 whitespace；forbidden 拒绝 `=`；allowed/forbidden 校验 unpadded unused bits；required 走
  provider strict last-chunk。cold exact-size validator 逐字符给出稳定 offset，但不进入正常 decode；
- 默认 strict route 与 peer 必须共同计入相同 policy scan。不能拿 simdutf 的 forgiving one-pass 数字判 HYLE
  输，也不能通过删除严格语义换性能；若 scan 自身 material，唯一允许的内部纠正是复用/扩展既有
  `container_text` bulk search，不改变 API 或默认值。

### 有限 FIX ledger

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| `G01-F01` | 没有公开类型、size 与 caller-buffer surface | 新 `base64.h`、header-inline conveniences、component header 宣告 | self-contained、compile/API、overflow/default tests |
| `G01-F02` | simdutf 能力未映射 HYLE 严格策略 | 在 `string_algo.cc` 或相邻单一 TU 实现 policy validator 与 provider adapter | RFC/敌意矩阵、typed progress、无复制 codec |
| `G01-F03` | 没有 consumer/document/delivery 路线 | 定向 unit、Container installed sample、architecture/user guide 与 provider/license 说明 | W1/WSL/Mac、source/static/shared、symbols/deps |
| `G01-F04` | 无同合同工程性能证据 | 新的窄 Base64 P1 contract/bench/evidence；只含 bulk image decode 与批量短 accept encode | direct simdutf semantic adapter、双顺序、5% NI、qualified bundle |

本表冻结后不增加 owning、streaming、MIME、mixed-auto、JSON/data-URI 或第二 provider。红队发现属于上述
四项的缺陷可作一次纠正；新能力必须退回 `SCOPED/AUDITED`。

## 七问（审计冻结）

1. 第三方：只借 vendored simdutf v9.1.0，版本/双许可证/安装已由 `container_text_simdutf` package
   dependency 锁定；无新下载，现有 scalar fallback 只覆盖 Unicode，不冒充 Base64 fallback。
2. 功能：四个推理基座、Folly 与 simdutf 共同需要的 contiguous bulk、两 alphabet、size、明确 validation
   已全部进入；owning/data URI/JSON 属于可组合高层，streaming/MIME 没有本轮准入。
3. 复用：使用 Core 编译/export 基础、Container 的 StringView/span façade、Mem/Container caller storage、
   Platform 驱动的既有 component/provider 构建；不新增假的 Error/Platform 调用。
4. 性能：复杂度最优 O(n)，size O(1)/O(n) 分层；P1 固定 256 KiB multimodal-style strict decode 与批量
   20-byte WebSocket-accept-style encode，计费相同 buffer/result/validation，peer 为同 revision direct
   simdutf semantic adapter，5% material margin。
5. FFI：项目当前没有 C consumer/Native FFI surface；本轮不恢复伪 C API。类型擦除跨语言边界仍由未来
   有真实 consumer 的 native FFI 决策处理。
6. 交付：新头属于 `container_text` split public header；同一 exported core 在 source/static/shared 可用，
   不泄漏 simdutf target/header；Container package probe 必须实际调用 encode/decode。
7. 易用性：默认安全、策略名不含糊；exact 与 max 两条 sizing recipe、partial-output 丢弃、非 NUL 输出、
   no-overlap、byte/char 转换和 forgiving 选择写入 Container guide。

Audit verdict 为 **PASS**：能力、API、provider、成本和有限改动集已冻结，允许进入产品实现；未运行的
correctness/delivery/P1 不在本阶段冒充通过。

## 产品实现回执

- `G01-F01/F02`：新增 `<hyle/container/base64.h>` 与 `string_algo.cc` 中唯一 provider adapter；公开面
  保持 caller-buffer、typed size/progress、显式四类策略。RFC 4648、all-byte、0..513 长度尾块、错误
  offset、padding/whitespace/alphabet、容量和 overlap 测试均通过。
- `G01-F03`：header self-containment、Container 安装样例和 source/shared/static relocated consumer
  已实际调用 codec；新头由中央 `container_text` split-header 集安装，simdutf 头仍不泄露给用户。
- `G01-F04`：`container-text-base64-p1-v1` 固定两个同合同 consumer shape、双执行顺序和独立 placement
  artifact。peer 使用锁定 revision 的外部 simdutf 声明，并链接产品内字节相同的 provider core；自定义
  validator 同时锁 benchmark/shape hash、语义 oracle、函数未折叠和 placement authenticity。
- P0：encoded/max size 为 O(1)，checked size/codec 为 O(n)；无 allocation、lock、atomic、TLS 或
  syscall。初次诊断发现默认 strict decode 的通用 policy loop 产生约 45% 固定 tax；把 required-padding
  常用路线与 unpadded tail 状态机算法分离后，双顺序诊断回到约 ±1.5%。语义、扫描阶段和 provider
  均未减少；该诊断不冒充 candidate P1。

候选前验证回执：

| 门 | 环境/形状 | 结果 |
|---|---|---|
| correctness/header | W1-WIN MSVC Release shared，定向 `container_text` | `2/2 PASS` |
| delivery | W1-WIN Container source/shared/static + relocated installed consumer | `PASS` |
| portability | W1-WSL GCC 14 Release shared，定向 header/unit | `2/2 PASS` |
| UB | W1-WSL GCC 14 RelWithDebInfo static + UBSan，定向 unit | `PASS`，零 sanitizer report |
| contract validator | W1-WIN primary + placement | semantic/same-contract/placement `PASS` |

M1-MAC、不可变 candidate P1 和独立终局红队仍待执行，因此本表不构成 `CLOSED`。

## Product candidate（`CANDIDATE`）

- immutable product commit：`e50a2d94f601ba0d9b7d87c854c868126ca4e596`；对应 tree
  `b6d00764459b8c9d9825a5e20b652e19aee9c585`。公开 API、codec/policy 算法、provider 边界、正常
  consumer shape 和常态成本从该提交起停止变化。
- candidate 前已在 W1-WIN 与 W1-WSL 通过定向 header/correctness，W1-WSL UBSan 无报告；W1-WIN
  source/shared/static 与 relocated consumer 均通过，正式合同的 semantic/same-contract/placement
  validator 也已通过。
- `AUDITED -> CANDIDATE` 的明确 previous revision 为
  `e50a2d94f601ba0d9b7d87c854c868126ca4e596`。后续只允许证据、task/user/developer 文档、测试专用
  修正和状态投影变化；发现产品/API/算法问题必须退回 `AUDITED`。
- 尚待 M1-MAC correctness、不可变候选正式 P1、终局红队、terminal evidence 与 LOC；未运行项不提前
  写成 PASS。

## 终局验证与裁决

candidate `e50a2d94f601ba0d9b7d87c854c868126ca4e596` 的产品 tree 为
`b6d00764459b8c9d9825a5e20b652e19aee9c585`。从 candidate 到终局，
`git diff e50a2d94..HEAD -- include/hyle/container/base64.h src/container_text/string_algo.cc
src/container_text/CMakeLists.txt cmake/HyleComponents.cmake` 为空；后续只增加状态空间测试、证据、task、
architecture 与状态投影，没有用 P1 点值反向改变 API、算法或正常 workload。

| 风险 / 命令或 receipt | 结果 |
|---|---|
| W1-WIN exact factor，MSVC Release shared | header + `container_text` `2/2 PASS`；RFC 4648、全部 byte、0..513 provider tails、策略/错误/容量/overlap 全过 |
| candidate 后独立状态空间红队 | 长度 0..5 的 271,453 个 adversarial strings × 12 个 alphabet/padding/whitespace 组合；cold/fast 接受性及成功 progress 全一致 |
| W1-WSL GCC 14 | Release shared `2/2 PASS`；RelWithDebInfo static UBSan `PASS`；candidate 产品不变后的 static ASan header/unit/ISA `3/3 PASS`，零 sanitizer report |
| M1-MAC MacBook Air arm64、AppleClang 21、exact candidate | header + `container_text` `2/2 PASS` |
| Container 三交付 | W1-WIN source embedding、relocated shared、relocated static consumer 全部 `PASS`；样例实际执行 Base64 round trip |
| shared boundary / provider | dylib 导出 Base64 encode/decode/checked-size；导出 simdutf 符号数 0，安装头无 simdutf 泄漏；双 license 由 `container_text_simdutf` package dependency 安装 |
| P1 `container-text-base64-p1-v1` | W1-WIN n=31；qualification/performance/verify `PASS`，两个 workload 的 8 格 5% NI 全过、无 `LOSE`；完整数字见 [`benchmarks-container-text-base64`](../../evidence/performance/benchmarks-container-text-base64.md) |
| 独立 bounded red-team | `PASS`；核对 malformed/policy、error precedence、capacity/overlap、partial output、公开易用性、simdutf mapping、symbols/license/delivery、benchmark 同合同与文档限制，无产品反证 |

七门终局结论：scope、architecture、correctness、delivery、usability_docs、red_team 与
engineering_performance 全部 `PASS`，`container_text` v3 可以签发 `CLOSED`。公开限制是 caller-owned
contiguous buffer、无隐式 allocation/NUL、input/output 不重叠、失败 output 不提交；streaming、MIME、
mixed alphabet、owning convenience、JSON/data-URI 与 WebSocket 状态机不属于 G01。

只有以下新事实可以重开：可复现的错误接受/拒绝、错误 bytes/progress、overflow/overlap/UB 或 provider
映射缺陷；真实消费者证明 contiguous bulk 之外存在两个 owner 重复同一通用 codec glue；公开 API、策略、
算法/provider、正常 workload、组件依赖、source/static/shared 交付或 simdutf revision material 改变；或者
合格同合同证据证明超过预声明 5% margin 的固定回归。一次地址/顺序点值或其他协议层功能不是 trigger。

`python scripts/loc-report.py --module container_text --commit
e50a2d94f601ba0d9b7d87c854c868126ca4e596 --self-check --json` 为 `PASS`：全库 531,562 行、自实现不含
测试 92,715 行、`container_text` 实现 1,580 行。
