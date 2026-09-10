# IO TLS v3：完整凭据来源

## Closure card（`SCOPED`）

- component / owner / round：`io_tls` / IO / `io-tls-v3-credential-sources`；
  kind=`REOPEN`；
- baseline commit：`758819bc7d53b9730bacbea7095b0f1faa2c149f`；前轮 terminal record 为
  [`io-tls-v2-duplex-progress`](io-tls-v2-duplex-progress.md)；
- trigger 与证据：已接受的基础能力扩展决策
  [`G07`](../../../architecture/foundation-capability-expansion.md#g07tls-credential-source) 证明当前仅接受
  file path 的 `TlsClientOptions` / `TlsServerOptions` 不能承载 secret-manager、嵌入配置或 encrypted PEM，
  用户只能绕过 HYLE 的 provider-neutral `TlsContext`；SGLang 的真实 TLS server 配置包含 encrypted-key
  password，Boost.Asio、Folly 与 cURL 均提供内存 credential 路线；
- 正常 consumer 任务：推理或低延迟服务从文件、配置中心或进程内 secret bytes 同步构造 client/server
  TLS context，可使用加密私钥；构造完成后立即释放路径、PEM、password 与 callback state，连接只持有解析后
  的不可变 provider context；
- 主产品纵切 / 具名来源：SGLang HTTPS server 的 key/password 配置，以及 llama.cpp/ggml、vLLM 和 ds4
  所需的 TLS termination/in-process client 组合；OpenSSL 是现有锁定 provider；
- 正交反偏置纵切：渲染资源流送与普通 embedded client 会从 application resource/config bytes 构造 CA 或
  client identity；它防止 API 被单一 Web server 的文件布局、reload policy 或框架 callback 形状特化；
- 跨模块 public journey 与每段 owner：`io_tls` 拥有 provider-neutral credential source、同步解析、secret
  lifetime、typed error 与 provider state；Error 承载失败，Container 提供 borrowed byte/string view，Mem
  继续只负责每连接 pump allocator；watch/rebuild/atomic swap 与 route/listener policy 属 `hyle-service`；
- scope：CA、certificate chain、private key 的 file path 与 borrowed PEM bytes；encrypted PEM 的 borrowed
  password 与 non-owning callback/context；client identity 和 server identity；mTLS trust source；同步构造期间
  的 callback/error/lifetime 合同；既有 context/stream ownership 与三种 C++ 交付；
- non-goals：credential watch/reload/atomic publication、cipher-string profile、provider engine、PKCS#11/HSM、
  certificate-selection callback、DER/PKCS#12、native provider handle、C/Native FFI、后台线程、另一 TLS
  provider，或修改 handshake/read/write/shutdown 数据路径；
- 直接依赖 / 直接消费者：现有 `core,error,io_core,mem,sched,container_text,time` 与 private OpenSSL 不变；
  直接消费者为 `io_http`、`io_http2`、`io_transfer` 和公开 IO package sample。预计不新增 HYLE component edge；
- 性能 tier 与原因：`P0`。credential parsing 是同步冷构造路径，成本应为输入字节线性、只含 provider 必需
  allocation；不得在 context 或每条 stream 留下 source/password/callback，也不得改变连接对象 layout、
  operation dispatch 或稳态数据路径。前轮 transport `P1` 继续绑定未变数据路径，不因冷配置扩面重跑；
- 适用机器 / 仪器 / material margin：W1-WIN、W1-WSL、M1-MAC 做正确性与交付；P0 通过复杂度、ownership、
  provider-call/code-path 与产品 diff 审计，不用纳秒计时，也不建立 material margin；
- 调查与纠正预算：一次 provider/API 审计、一个产品批次、一次 correctness/red-team 产品纠正；没有 P1
  tuning 轮。若改变公开形状或 provider 算法，退回 `AUDITED`；
- 预声明验证：public-header self-containment；file/PEM 等价；单证书和完整 chain；client/server/mTLS
  handshake；未加密和加密 key 的 borrowed password/callback；错误 password、callback failure、缺失配对、
  empty/malformed/truncated/NUL/oversize 输入；callback 不保留、秘密不进入错误；OOM/清理；W1-WIN/W1-WSL/
  M1-MAC；WSL ASan/UBSan；IO source/static/shared relocated consumer；OpenSSL 类型/符号/头不泄漏；终局红队；
- stop / blocker 条件：若稳定 OpenSSL API 无法在不保留 callback/secret 的情况下统一 file/PEM 语义，或三平台
  provider 差异要求泄漏 provider-native type，则以精确 blocker 终止；不得降低证书验证默认值、调用交互式
  terminal password prompt、复制 TLS runtime，或修改连接热路径来伪造完成。

## SCOPED 不变量

1. 所有 credential input 只在 `make_client()` / `make_server()` 同步调用期间借用；成功或失败返回后 HYLE
   不保留 source、password、callback/context 或指向它们的指针。
2. 构造成功的 `TlsContext` 只拥有解析后的 provider state；既有 `TlsStream` 继续保活旧 context。reload
   属上层显式构造新 context 后原子替换，不进入本轮 API。
3. public API 只表达 HYLE 语义，不暴露 `SSL_CTX`、`BIO`、`X509`、OpenSSL callback signature、filetype 或
   error queue。文件与内存来源可以组合，但无效 identity 配对必须 typed fail。
4. 没有 password 时，encrypted key 必须 typed fail，绝不能让 provider 从 stdin/terminal 交互读取。错误
   消息不得包含 password、PEM 内容或 callback 返回内容。
5. 本轮不得改 TLS stream 数据路径、每连接对象字段或默认 verification/security 语义；因此不存在合法的
   credential microbenchmark 或重跑 transport P1 的理由。

## 进入 `AUDITED` 前的有限审计队列

- 对照 OpenSSL、Boost.Asio、Folly 与 cURL，冻结一个 provider-neutral source value 和 password
  borrowed/callback value；明确 none/file/PEM 的非法状态、默认值、组合和命名；
- 证明 file 与 length-delimited memory 的 certificate chain、trust CA 和 private-key loader 可以共享一套
  生命周期/error 语义；冻结 chain 数量、embedded NUL、超大输入、duplicate CA 与 error precedence；
- 冻结 password callback 的 HYLE signature、调用次数、临时返回 view、失败传播和 provider adapter；证明
  callback 不安装到 `SSL_CTX` 且构造返回后不可再调用；
- 从全部现有 TLS 公开入口、测试、bench、sample 和直接消费者反向映射旧 `*_file` 字段；项目不承诺兼容，
  本轮选择单一自然形状并完整迁移，而非长期保留两套配置；
- 冻结有限 `KEEP/FIX/BORROW/REJECT` ledger、correctness 矩阵、P0 结构门、三平台和三交付范围。完成这些
  之前不得修改产品代码。

## 能力与形状审计（`AUDITED`）

| 能力族 | 层级 | 准入旅程/来源 | 语义与生命周期 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|---|
| trust authority file/PEM | 原子机制 | client system/explicit trust、mTLS；Asio/cURL | source 仅在同步构造借用；context 拥有解析后 store | 只有 file；client none=system trust | `PARTIAL` | `BORROW + FIX` |
| identity certificate chain | 原子机制 | HTTPS server、mTLS client；Asio/Folly | leaf + 最多 63 个 intermediates；file/PEM 同 parser | 只有 OpenSSL chain file | `PARTIAL` | `BORROW + FIX` |
| private key file/PEM | 原子机制 | server/client identity；Asio/Folly/cURL | 与 cert source 可混用，解析后 provider 拥有 key | 只有未加密 PEM file | `PARTIAL` | `BORROW + FIX` |
| encrypted-key password | 原子机制 | SGLang `ssl_keyfile_password`、secret manager | borrowed value 或可失败同步 callback；不安装进 context | 无 HYLE 路线；provider 默认可能交互提示 | `MISSING` | `BORROW + FIX` |
| identity pair + typed construction | 复合功能 | 普通 `make_client/make_server` | 配对、trust、version、ALPN、allocator 一次校验并原子提交 context | file-only pair 已有 | `PARTIAL` | `KEEP + FIX` |
| context lifetime / stream retention | 复合功能 | 连接跨 reload 生存 | context immutable/ref-counted，stream 保活旧 provider state | 已完整 | `COMPLETE` | `KEEP` |
| credential reload / atomic swap | 应用策略 | certificate rotation | service owner 监控、重建并替换 context | `hyle-service` owner 尚未施工 | `ROUTED` | 不进入连接或本轮 |
| DER/PKCS#12、cipher/provider/HSM、cert selection | 策略/扩展格式 | 尚无两个 provider-neutral consumer | 会引入格式、硬件或安全 profile policy | watchlist | `OUT_OF_SCOPE` | `DEFER/REJECT` |

公开面反向追踪覆盖 `TlsClientOptions`、`TlsServerOptions`、`TlsContext`、`TlsSession`、`TlsStream` 与 TLS
operations。stream/session/ALPN/version/verification/allocator 每项仍映射到 v1/v2 已准入旅程，没有
benchmark-only 入口；只有四个 `*_file` 字段把 credential source 锁成路径且缺少 password。项目未发布且
不承诺兼容，本轮删除这些旧字段、完整迁移 HYLE tests/bench/sample/consumer，只保留一个自然配置面。

### API / usability freeze

公开头继续为 `<hyle/io/tls.h>`，新增的 provider-neutral 形状为：

```cpp
enum class TlsCredentialSourceKind : std::uint8_t { kNone, kFile, kPem };

class TlsCredentialSource {
public:
  constexpr TlsCredentialSource() noexcept = default;
  static constexpr TlsCredentialSource file(container::StringView) noexcept;
  static constexpr TlsCredentialSource pem(container::StringView) noexcept;
  constexpr TlsCredentialSourceKind kind() const noexcept;
  constexpr container::StringView value() const noexcept;
};

using TlsPasswordCallback =
    util::FunctionRef<Result<container::StringView>() noexcept>;

enum class TlsPrivateKeyPasswordKind : std::uint8_t {
  kNone, kBorrowed, kCallback
};

class TlsPrivateKeyPassword {
public:
  constexpr TlsPrivateKeyPassword() noexcept = default;
  static constexpr TlsPrivateKeyPassword
  borrowed(container::StringView) noexcept;
  static TlsPrivateKeyPassword callback(TlsPasswordCallback) noexcept;
  constexpr TlsPrivateKeyPasswordKind kind() const noexcept;
};
```

`TlsClientOptions` 使用 `certificate_authority`、`certificate_chain`、`private_key`、
`private_key_password`；`TlsServerOptions` 使用 `certificate_chain`、`private_key`、
`private_key_password`、`client_certificate_authority`。调用形状例如：

```cpp
TlsServerOptions options;
options.certificate_chain = TlsCredentialSource::pem(certificate_pem);
options.private_key = TlsCredentialSource::file("server-key.pem");
options.private_key_password =
    TlsPrivateKeyPassword::borrowed(secret_password);
auto context = TlsContext::make_server(options);
```

- source value 是无 allocation 的 borrowed tagged value；default `kNone` 与显式 `file("")` / `pem("")`
  不混淆。file path 拒绝 empty、embedded NUL 和不可表示长度；PEM 是 length-delimited text，拒绝 empty、
  embedded NUL 和 `BIO_new_mem_buf` 的 `INT_MAX` 外输入；
- server identity 必须有 cert+key，client identity 要么两者皆无、要么两者皆有；cert/key 可分别来自 file
  或 PEM。password 无 key、verification 关闭却提供 CA、server mTLS 开启却没有 CA 都是
  `InvalidArgument`，不静默忽略配置；
- client `verify=kPeer` 且 CA 为 none 继续使用系统 trust；server mTLS 必须显式提供 client CA。CA bundle
  可有多个证书，重复证书等幂接受；identity chain 最多 64 张，防止错误 bundle 造成无界 provider state；
- callback 在调用 `make_*` 的线程串行调用零次或多次；只读 private key，因此不暴露 OpenSSL 的
  read/write purpose。它可以返回任意 `Status`，该失败原样传播；成功 view 必须由 callback 外部状态拥有，
  并至少活到 `make_*` 返回。callback 本身也须活到 `make_*` 返回；
- borrowed/callback password 可以为空且按长度处理；过长、non-null size 配 null data 或 provider buffer
  不足 typed fail。没有 password 也始终安装一次栈上拒绝 adapter，encrypted key 绝不回退到 provider 的
  stdin/terminal prompt；
- password adapter 只传给 `PEM_read_bio_PrivateKey`，不调用 `SSL_CTX_set_default_passwd_cb*`。返回后
  context/stream 不含 credential source、password 或 callback；错误文本不拼接 secret/PEM/provider queue；
- `FunctionRef` 是现有 `util_core` 的恰当同步 borrow，因此新增 `io_tls PUBLIC_DEPS util_core`，不复制第二套
  erased callback。它允许 lvalue capturing lambda、函数和 compile-time callable，不分配；guide 明确禁止
  把短命 callable 存进长期 options 后再调用；
- file 与 PEM 都使用 OpenSSL BIO + 同一 chain/key/CA loader；provider objects 采用 RAII 临时 owner，成功
  后由 `SSL_CTX` 增持，任意失败只销毁未发布 context。OpenSSL 类型不出 public header。

### 复杂度与性能合同

- source/tag/password option 的创建为 O(1)、零 allocation；PEM 构造不复制。file source 若 view 没有已知
  NUL terminator，可产生一个仅活到构造返回的 O(path bytes) 临时副本；
- credential parse 为 O(input bytes)，certificate/key/CA 的 allocation 只来自 OpenSSL provider object；
  identity chain 占 O(chain certificates)，CA store 占 O(unique CA certificates)；
- 新类型只存在于 cold options/context factory；`ContextImpl`、`TlsStream`、pump scratch、TLS operation 和
  transport syscall 均不得改变。P0 通过产品 diff、对象 layout/static assertions 和 provider call graph
  证明零连接热路成本；不计时、不建立 credential benchmark；
- 旧 v2 transport P1 只在 stream/operation/provider transport 路径发生产品变化时失效。本轮预声明不触碰
  这些路径，因此保留其历史证据但不冒充新 credential 性能排名。

### 有限 FIX ledger

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| `G07-F01` | 四个路径字段不能表达 memory/password，且 callback 形状不应重复实现 | 新 source/password value；`io_tls -> util_core`；删除旧字段并迁移全部 HYLE consumer | header/API misuse、component graph、无 provider leak |
| `G07-F02` | file-only provider shortcuts 不能统一 secret lifetime，默认 key loader可能交互 prompt | BIO source、chain/CA/key RAII loader、栈上 password adapter、typed validation/error | file/PEM/mixed、encrypted key、callback failure、无 retain/prompt |
| `G07-F03` | credential correctness/安全边界无覆盖 | 扩展 io_tls tests 与必要 package sample；只做测试 fixture/adapter | chain/CA/mTLS、坏输入/密码、callback次数与销毁、ASan/UBSan、三平台 |
| `G07-F04` | 旧 guide 声明 memory/encrypted unsupported，构建最低版本和新 public edge需校准 | user/architecture/setup、OpenSSL 3.0 configure floor、三交付/符号/许可证检查 | source/static/shared relocation、TLS OFF、docs/red-team/P0 |

本 ledger 到此冻结。不得加入 reload、cipher、HSM、DER/PKCS#12、connection callback、第二 provider 或
TLS transport tuning；新事实若改变公共形状或范围必须退回 `SCOPED/AUDITED`。

## 七问（审计冻结）

1. **第三方：** OpenSSL 3.x 仍是唯一 private provider，由宿主 SDK/dynamic security update 路线提供；
   Windows vcpkg、WSL system package、macOS Homebrew 均有现有安装合同。产品不 vendoring crypto；
   Boost.Asio `cb94f33f`、Folly `65749da4` 与 cURL `01346829` 只作当前 source/password peer。
2. **功能：** file/PEM trust、chain、key、borrowed/callback password、client identity、server identity、mTLS
   和 context lifetime 覆盖 G07 的推理与正交 embedded journey；reload 路由 `hyle-service`，其它格式/策略
   无稳定 MUST。
3. **内部复用：** Status/Result、StringView、Allocator、TLS transport 均沿既有 owner；同步 callback 改用
   `util_core::FunctionRef`。OpenSSL 独占证书/key/parser 决策，不在 HYLE 复制密码学或 PEM parser。
4. **性能：** 最优线性 parse、无 PEM copy、无长期 secret/callback；只允许临时 path C-string copy。
   stream layout/data path 不变，P0 足够且禁止苹果对橘子的 credential 纳秒比较。
5. **旧 ABI：** 项目没有 C/Native FFI，本轮删除未发布旧 C++ file-only 字段，不加兼容 alias；OpenSSL C
   API 只是 private provider boundary。
6. **构建交付：**中央图新增真实 `util_core` public edge；OpenSSL feature 继续 `AUTO/ON/OFF`，但 configure
   与 installed dependency 统一要求 3.0；必须验证 source/static/shared、relocation、TLS OFF 和 public header。
7. **易用性：** 一个 source value 覆盖 file/PEM、一个 password value覆盖 direct/callback；默认安全、无效
   组合尽早 typed fail。guide 给 file、memory、encrypted 和 reload 四条 recipe，并写 callback/borrow 陷阱。

审计结论为 **PASS**：能力边界、自然 C++ 形状、provider 映射、依赖、有限改动集与 P0 门已冻结，允许进入
产品实现；correctness、三平台、三交付、红队仍未运行，不能提前写成通过。

## 产品实现与候选（`CANDIDATE`）

- `G07-F01/F02`：`<hyle/io/tls.h>` 的单一 source/password 配置面与 OpenSSL BIO/RAII loader 已完成；
  file、borrowed PEM、混合 cert/key、encrypted key、borrowed/callback password、client/server identity、
  mTLS 和 64 张 chain 上限均走同一同步构造与 typed failure 路线。旧 `*_file` 字段已从 HYLE
  tests、bench、sample 与直接 consumer 清除，没有兼容别名。
- `G07-F03`：Windows MSVC Release 与 WSL GCC 14/OpenSSL 3.0.13 的 header/unit 均为 `2/2 PASS`；
  边界矩阵覆盖 empty/missing/malformed/truncated/NUL/oversize、错误密码、callback failure/不保留、
  duplicate CA、完整 chain、实际 encrypted-key handshake 与全内存 mTLS。
- `G07-F04`：Windows 的 TLS-on source embedding、relocated shared SDK、relocated static SDK consumer
  均通过；shared DLL 没有导出 OpenSSL/X509/BIO/EVP 类型符号，static package 明确生成
  `OpenSSL|3.0` dependency spec。通用 TLS-off IO smoke 也通过，但不冒充 TLS-on 交付证据。
- P0：source/password value 为 O(1)、零 allocation；credential parse 为 O(input bytes)，只有
  OpenSSL provider 对象和非 NUL-terminated file path 可能分配。相对 baseline 的忽略空白产品 diff
  只触及 credential helper 与 `make_client/make_server` 配置，`ContextImpl`、`TlsStream` layout、pump、
  handshake/read/write/shutdown 路径未改变，因此没有连接热路新增成本，也不运行 credential 纳秒 benchmark。

immutable product commit 为 `4bd86adf0b4027414ebe5cb576b217a5e87c2170`，tree 为
`706e74449022ec825c7e5432cbbaf53815092112`。从此公开 API、provider 算法、正常 consumer shape 与成本
结构停止变化；发现产品问题必须退回 `AUDITED`。`AUDITED -> CANDIDATE` 的 previous revision 是该产品
提交自身。尚待候选绑定的 WSL ASan/UBSan、M1-MAC、直接 HTTP/HTTP2/Transfer consumer、组件合同复验、
终局独立红队、LOC 与 terminal record；这些未运行项不提前写成 PASS。

## 终局验证与裁决

候选 `4bd86adf0b4027414ebe5cb576b217a5e87c2170` 的产品 tree 为
`706e74449022ec825c7e5432cbbaf53815092112`。候选状态提交之后，
`git diff 4bd86adf..HEAD -- include/hyle/io/tls.h src/io_tls cmake/HyleComponents.cmake
cmake/HyleIoDependencies.cmake` 为空；后续验证没有反向修改 API、provider 算法、对象 layout 或普通数据路径。

| 风险 / 候选绑定 receipt | 结果 |
|---|---|
| W1-WIN / MSVC 19.50 / Release shared | `io_tls` header + unit `2/2 PASS`；显式开启 TLS 与 HTTP/2 后，`io_tls`、`io_http`、`io_http2`、`io_transfer` 的 unit/header `9/9 PASS` |
| W1-WSL / GCC 14.2 / OpenSSL 3.0.13 | Release shared header + unit `2/2 PASS`；独立 ASan 与 UBSan RelWithDebInfo 构建各 `2/2 PASS`，零 sanitizer report |
| M1-MAC / MacBook Air `Mac16,12` / AppleClang 21 | 产品 candidate 的 Git archive SHA-256 为 `D4DEEB3FDCBA513084416485414B5E0DEFA1B0AB9F146FC285A625BDB25B077E`；远端核对同 SHA 后，OpenSSL 3.6.1 header + unit `2/2 PASS` |
| source / shared / static delivery | W1-WIN TLS-on source embedding、relocated shared SDK、relocated static SDK consumer 全部 `PASS`；static config 生成 `OpenSSL|3.0` dependency spec |
| feature / component contract | candidate 后组件合同 `PASS`；TLS-off exact IO factor 仍配置、构建和测试，且不探测/链接 OpenSSL；显式 TLS-on 缺 OpenSSL 3.0+ 会配置失败 |
| shared provider boundary | `hyle_io_tls.dll` 导出表中 OpenSSL/SSL_CTX/X509/EVP/BIO 名称匹配数为 `0`；动态依赖 OpenSSL runtime，但公开头与 sample 的 provider type 匹配数为 `0` |
| P0 成本结构 | source/password value 构造 O(1)、零分配；PEM 不复制，file path 仅在不能证明 NUL 结尾时做一次 O(path bytes) 临时复制；解析 O(input bytes)，连接 layout、pump 与 handshake/read/write/shutdown 路径相对 baseline 不变 |

Mac 上 Homebrew OpenSSL 3.6.1 由 macOS 26 构建，而 HYLE source configure 保留 14.4 deployment target，
linker 因此给出 dependency deployment warning；实际 macOS 26.6.2 运行通过。本项只签当前 M1-MAC 的
source correctness/portability，不冒充 OpenSSL bottle 在 macOS 14.4 的二进制兼容证明，也不改变
Windows 三交付结论。

### Bounded red-team receipt

- 全部 HYLE TLS 公开入口、tests、bench、sample 和直接 consumer 已反向扫描；旧 HYLE `*_file` 字段为零，
  唯一文字命中是 Asio/OpenSSL peer 自己的 file API。file/PEM/mixed source、完整 chain、client/server
  identity、mTLS、encrypted key 与 password callback 均有正常 journey，不存在 benchmark-only surface；
- source、password 和 callback 只由 `make_client/make_server` 同步借用。callback adapter 是栈对象，
  只传给一次 private-key parse，不调用 `SSL_CTX_set_default_passwd_cb*`；构造后销毁 context、执行 handshake
  都不会重新调用 callback。错误文本不拼接 PEM/password/provider queue；callback 主动返回的 `Status`
  按公开合同原样传播，由 callback owner 负责其消息内容；
- BIO、X509、EVP_PKEY 使用唯一 RAII owner；成功安装由 `SSL_CTX` 增持，任意中途失败最终释放整个未发布
  context。HYLE metadata/path allocation 使用 `nothrow` 并返回 `ResourceExhausted`；provider error queue 中
  的 `ERR_R_MALLOC_FAILURE` 同样映射为 `ResourceExhausted`。没有采用进程全局且要求在 OpenSSL 首次使用前
  安装的 allocator hook 来伪造局部 OOM；真实 malformed/password/config 失败矩阵与 ASan leak 检查覆盖清理；
- empty、missing、malformed、truncated、embedded NUL、`INT_MAX` 外输入、65 张 identity chain、duplicate
  CA、错误/缺失/过长密码、callback failure、忽略型配置和无效 allocator 均 typed fail。duplicate CA
  等幂接受；没有密码的 encrypted key 使用拒绝 callback，不会退回 terminal prompt；
- `io_tls -> util_core` 是公开头使用 `FunctionRef` 所需的真实边；OpenSSL 仍是 private provider，source、
  static 与 shared 的 dependency/compile-definition/安装闭包一致。未发现复制 Error、Container、Mem、
  Sched 或 IO Core 能力，也没有 C/Native FFI、稳定 C++ ABI 或兼容旧字段；
- 本轮是冷配置能力，未运行 credential 纳秒 benchmark。没有拿 file I/O 与 memory parse、未加密与加密 key
  或不同验证语义做苹果/橘子排名；已有 transport P1 不因未变数据路径重新运行或冒充本轮成绩。

七门终局结论：scope、architecture、correctness、delivery、usability_docs、red_team 与
engineering_performance 全部 `PASS`，`io_tls` v3 可以签发 `CLOSED`。当前公开限制是同步构造、借用输入、
file/PEM 两种来源、identity chain 最多 64 张；credential watch/reload/atomic swap 属 `hyle-service`，
DER/PKCS#12、provider/HSM、cipher profile 与证书选择策略仍无足够通用准入证据。

只有以下新事实可以重开：可复现的 credential parse、identity/mTLS、秘密泄漏、交互 prompt、OOM/清理、
callback lifetime 或 provider 错误；真实 consumer 证明当前 file/PEM/password 之外存在两个 owner 重复同一
通用 TLS glue；公开 API、provider 算法、OpenSSL minimum、对象 layout、依赖或 source/static/shared 交付
发生变化；或者 credential 改动进入连接热路径。未限定 benchmark 点值、应用 reload policy 或想恢复旧
字段都不是 trigger。

`python scripts/loc-report.py --module io_tls --commit
4bd86adf0b4027414ebe5cb576b217a5e87c2170 --self-check --json` 为 `PASS`：全库 532,574 行、自实现不含
测试 93,093 行、`io_tls` 实现 1,528 行。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
