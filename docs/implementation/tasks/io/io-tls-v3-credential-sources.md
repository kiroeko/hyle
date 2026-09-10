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

