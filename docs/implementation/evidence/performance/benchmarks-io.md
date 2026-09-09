# io 实测铁证


> 本页当前签发 `io_core-v1`、`io_tls-v2-duplex-progress`、`io_http-v2-duplex-progress`、
> `io_http2-v2-owner-progress-cost`、`io-transfer-v1` 与 `io-watch-v1` 的工程性能结论。`io_tls-v1`、
> `io_http-v1` 与旧 io 聚合数据均是
> 历史事实，不再是对应活动轮的当前性能签名；当前结论也不能扩大成完整服务器、
> 全平台或全操作 SOTA。

## 2026-09-10 `io-watch-v1` 当前工程证据（`PASS`）

产品 candidate/tree 为 `30e133dbcf37e806f9deb1b121cdf0ae559546f6` /
`b0df48f10774ae119b03fc1e779ababfe3b5e744`；peer 是 source-pinned libuv
`3e606bc30a842701f8d12007d3ade662d258b76b`。三平台使用同一个
`9d1305102cbb2d4ab619856dca91d54fd974c29993dce00082008d4b316ac125` 合同：双方都从已经建立的
watcher 开始计费，到 consumer 收到并分类同一批事件结束；都包含各自的 event-loop/task dispatch、
atomic replace 或 directory burst、事件校验和 teardown。Windows 发布者对两侧使用同一个最多 50 次、
每次 1 ms 的瞬时 sharing/access retry，并把 retry 计入时间；它模拟正常的可靠发布者，不改变 watcher
语义或完成条件。因此这里没有把 raw syscall、callback 或较少语义的队列同完整 HYLE 生命周期作
苹果/橘子比较。

advantage 为 HYLE/libuv op/s，越大越有利于 HYLE。primary 与 placement、candidate-first 与
peer-first 都独立报告；Windows/macOS 为每格 15 个独立进程，WSL 按预声明精度升级到 31 个：

| machine / workload | HYLE / libuv op/s 范围 | advantage / 置信区间范围 | exact | closure NI |
|---|---:|---:|---|---|
| W1-WIN exact-file atomic replace | 1,474.88--1,501.21 / 844.45--937.31 | 1.61588--1.76269 / lower 1.49491--1.57686 | 4/4 `WIN` | 10% `PASS`；严格 5% 亦全过 |
| W1-WIN non-recursive directory burst | 4,791.89--4,957.59 / 1,259.37--1,283.68 | 3.69678--3.89063 / lower 3.43010--3.63301 | 4/4 `WIN` | 10% `PASS`；严格 5% 亦全过 |
| W1-WSL exact-file atomic replace | 18,956.02--19,433.31 / 19,325.27--19,583.35 | 0.96813--0.98638 / lower 0.94623--0.97570 | 2 `PARITY`、2 `INCONCLUSIVE`、0 `LOSE` | 10% `PASS`；严格 5% 为 2 `PASS`、2 `UNRESOLVED` |
| W1-WSL non-recursive directory burst | 63,264.42--64,099.34 / 59,063.84--60,046.83 | 1.06254--1.06909 / lower 1.04309--1.05844 | 2 `WIN`、2 `INCONCLUSIVE` | 10% `PASS`；严格 5% 全过 |
| M1-MAC exact-file atomic replace | 21.23--21.26 / 21.23--21.26 | 0.99953--1.00000 / lower 0.99484--0.99859 | 4/4 `PARITY` | 10% `PASS`；严格 5% 亦全过 |
| M1-MAC non-recursive directory burst | 153.74--153.78 / 153.75--153.79 | 0.99987--1.00000 / lower 0.99598--0.99890 | 4/4 `PARITY` | 10% `PASS`；严格 5% 亦全过 |

closure card 在任何采样前把工程 material margin 预声明为 10%，其非劣下界是 `1/1.10 = 0.90909`；
三平台所有区间均在该界内，所以 terminal engineering gate 为 `PASS`。统一 runner 的合同使用了更严格的
默认 5% 门；它在 WSL exact-file 的两个 placement 格因下界 `0.94810/0.94623` 略低于 `0.95238` 而诚实
返回 `UNRESOLVED`，不是 `LOSE`。这项附加诊断不被隐藏，也不事后替换原先的产品行动阈值。

WSL 的 exact-file 路线显示 HYLE 约 1.4%--3.2% 的点差：HYLE 为完整通用路径支付 UTF-8 校验、multiple-target
expected-O(1) route、bounded coalescing 与 fiber wake，libuv adapter 可直接比较一个 basename 并回调；
同一实现到了 directory burst 可摊销这些成本，点估计反而领先约 6.3%--6.9%。Mac 两种形状均 parity，
Windows 两种形状均 win；没有跨平台、跨顺序的 material 固定劣化，也没有依据为了点值给对象增加
常态 metadata 或扭曲公开 API。single-owner、one outstanding `next()`、borrowed batch、显式容量和
显式 close/rescan 是换取无逐事件分配、无 busy polling 和 bounded teardown 的公开性能限制。

三份 bundle 的 A/A、5%/10% known-delta、timer floor、placement、artifact/build/consumer/source/peer
provenance 和采样稳定性全部 `PASS`；origin verifier 均为 `QUALIFIED`、`paths_verified=true`、
`provenance=EXACT`、`environment_policy=CURRENT`：

- [`W1-WIN bundle`](../../../../evidence/performance/io-watch-p1-v1-30e133db-w1-win-4c06445c/)：manifest / summary /
  controls SHA-256 为 `fcf7cb653c2ba2188b52629ff9922dc66063f4ed12c7a2da3a7460180f62d29a` /
  `d228f8b4e258fa0dc116e62d144dd57135047ed0ff8534e82506d62fcb5cfc03` /
  `a4e19c49fecf550232782e68ed504962cf7494e9c81682b4cd186eddc6244d56`；
- [`W1-WSL bundle`](../../../../evidence/performance/io-watch-p1-v1-30e133db-w1-wsl-4c06445c/)：
  `035ca99e7fb3d05d3077bacc78b9cbd31a8bdf8aba353059fedb02ea8aaef5ae` /
  `7224734b9cab7ac5c45bbbbb88ccc42b3787a7c9209827cec4df01b077339f4c` /
  `24035f0bb766814902aa632d76d08bb5ac0f93d1837deb0e7f4e007f18ba5dd1`；
- [`M1-MAC bundle`](../../../../evidence/performance/io-watch-p1-v1-30e133db-m1-mac-4c06445c/)：
  `9c1db2fac23bf5d4125a259aa72fa599094883bbe0ca8b7be2550a9a5bab040e` /
  `ac2ab1da37e402621229e7e910c87aeba5faae2707c85f0bf861ad5eabd23844` /
  `0bbe1c16b76ebde59097646af864d296785b0b9598c34f5fd28f2ce9be224222`。

首次 Windows attempt 在采样前暴露双方共用的 publish fixture 缺陷并中止；对称修复后才产生上述唯一
当前 Windows bundle。一次误用 `--formal` 的运行只因机器 profile 没有 formal toolchain 而在 build-contract
阶段失格，不承担产品结论。这里签发的是 exact-file replace 与单层 directory burst 的工程结论，不是
递归/network filesystem、所有 workload、完整 reload service 或 Formal SOTA 证书。

## 2026-09-09 `io_core-v1` 当前工程证据（`PASS`）

产品 candidate/tree 为 `ffa70e35cd7b7948990c8f04117acf3c6c209c8a` /
`4fd9be1c69dab36355401c128eb8eec2337f756e`；peer 是 source-pinned Boost.Beast
`cb94f33fcb13e90922905c2d8105508e71fa6db4`。候选后的 `3c498953` 只把性能 fixture 从已封闭的 raw
handle 构造改为公开 `TcpStream::adopt`，并更新候选记录；从产品 candidate 到该 harness commit 的
`include/hyle/**`、`src/**`、CMake 和依赖 diff 为空。第一次尝试在 configure/build 阶段暴露此 fixture
漂移，没有采样也没有 bundle；修正 fixture 后只执行一次资格化 P1。

双方都通过完整 public transport + HTTP/1 路径，在同一个 persistent worker 上处理相同 request、
streaming response、body validation 和 connection lifecycle；没有把 raw submit 与完整 peer、预建结果与
运行时解析或不同 payload 形状作苹果/橘子比较。advantage 为 HYLE/Beast op/s，每个顺序 15 个独立进程，
置信水平 99.26%，固定 non-inferiority margin 为 5%：

| workload / order | HYLE / Beast op/s | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| 128x1 KiB REST/stream / candidate-first | 62,343.81 / 61,583.79 | 1.00083 / [0.97501, 1.01460] | `PARITY` | `PASS` |
| 128x1 KiB REST/stream / peer-first | 62,298.59 / 62,000.69 | 1.00253 / [0.97934, 1.01621] | `PARITY` | `PASS` |
| 16x64 KiB fragmented / candidate-first | 17,739.87 / 5,837.15 | 3.03820 / [2.98619, 3.11586] | `WIN` | `PASS` |
| 16x64 KiB fragmented / peer-first | 17,714.89 / 5,901.75 | 3.00646 / [2.96475, 3.07418] | `WIN` | `PASS` |

A/A、5%/10% known-delta、timer floor、artifact/build/consumer/source/peer provenance 与采样前后稳定性全部
`PASS`；严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、`provenance=EXACT`、
`environment_policy=CURRENT`。唯一 68-file bundle 位于
[`io-core-p1-v1-ffa70e35`](../../../../evidence/performance/io-core-p1-v1-ffa70e35/)；manifest、summary、
controls SHA-256 依次为 `57927A806E2B91348A800ABCF2B7E84737263D3C9553EF304080A6BFAEAA8438`、
`9500EC766ADD7404B7C1ED260F0AFBD6EFA7FF61A7A95923125634EBE24D360F`、
`3C0DB58EB271CBFC4FF9612B7A7A84A149CEA6D8AB39504A61F5F64320AF3EC5`。

这是 W1-WSL、GCC 14.2、Release/static、portable ISA、单 worker 的完整 HTTP consumer 工程证据：它证明
核心 transport 没有给正常组合路线增加 material 税，并证明两个不同粒度没有固定输项；它不是裸
`io_core` 每个操作、WAN、多 worker、全部 backend、完整服务器或 Formal SOTA 证书，也不把 64 KiB
路线约 3 倍的优势外推到整个模块。

## 2026-09-09 `io_transfer-v1` 当前工程证据（`PASS`）

不可变 candidate/tree 为 `18bf62f81a91975e0776439d88fca3891f10e560` /
`98632557e0237bac0bed3ca5521e59420d89176e`；peer 是 source-pinned libcurl 8.22.0
`01346829096c61b372692f6dc43ffa778c6caccd`。本合同不是拿裸 copy 和可靠工作流比较：本地行双方都
传输 256 MiB、增量 FNV-1a 校验、前后复验源身份、同步并关闭 sibling temp、原子替换旧目标；HTTP 行
双方都完成 64 MiB、在 32 MiB 后中断、保留 partial、第二连接发送 Range/If-Range、验证 206/ETag/
Content-Range、同步并原子发布。runtime/server/fixture 在 timing 外，attempt state 与 teardown 在 timing 内。

advantage 为 HYLE/libcurl op/s，越大越有利于 HYLE；每个顺序 15 个独立进程，置信水平 99.26%：

| workload / order | HYLE / libcurl op/s | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| verified atomic local 256 MiB / candidate-first | 3.29 / 2.98 | 1.09797 / [1.06061, 1.12615] | `WIN` | `PASS` |
| verified atomic local 256 MiB / peer-first | 3.28 / 3.10 | 1.06796 / [0.95455, 1.10559] | `INCONCLUSIVE` | `PASS` |
| interrupted HTTP Range resume 64 MiB / candidate-first | 34.90 / 29.37 | 1.20629 / [1.13160, 1.25449] | `WIN` | `PASS` |
| interrupted HTTP Range resume 64 MiB / peer-first | 33.63 / 28.03 | 1.24736 / [1.13803, 1.27406] | `WIN` | `PASS` |

本地 publish 聚合 exact verdict 因一个顺序的 CI 跨 superiority 边界而诚实记为 `INCONCLUSIVE`，但两顺序
固定 5% 非劣门都通过；resume 双顺序明确 `WIN`。A/A、5%/10% known-delta、timer floor、artifact/build/
consumer/source/peer provenance 与采样前后稳定性全部 `PASS`；n=15 已裁决，没有触发 n=31。唯一有效
98-file bundle 位于
[`io-transfer-p1-v1-18bf62f8`](../../../../evidence/performance/io-transfer-p1-v1-18bf62f8/)；manifest、
summary、controls SHA-256 依次为
`BB150DE26D07691739C3158C764E79C746AAB84E4ACA36BB8E1C8B179C75769F`、
`81B519ACA75D129A0B50CD0C0F1D066AC07AF11D8E8F15797E561C65987945D5`、
`D8A4DAD9AE5A13F831E50BBC0FFE77E160CC8FF67D7A679B3B8AD64B3F8FBB5D`。严格 verifier 返回
`QUALIFIED`、`paths_verified=true`、`provenance=EXACT`、`environment_policy=CURRENT`。

这是 W1-WSL、GCC 14.2、Release/static、portable ISA、单 worker 的工程 P1，不是远程 WAN、多 worker、
所有 filesystem、所有 checksum、object-store、repository client 或 Formal SOTA 证书。结果只支持冻结的
可靠发布/续传合同，不外推为所有传输场景的普遍倍率。

## 2026-09-09 `io_http-v2` 当前工程证据（`PASS`）

不可变 candidate/tree 为 `18bf62f81a91975e0776439d88fca3891f10e560` /
`98632557e0237bac0bed3ca5521e59420d89176e`；peer 是 Boost.Beast
`cb94f33fcb13e90922905c2d8105508e71fa6db4`。本轮增加的 immediate progress seam 仅由 HTTP/2
组合路径使用；普通 HTTP/1 parser、message/body API、runtime、socket、输入、validation、teardown、
双顺序和固定 5% margin 均保持冻结合同。双方在一个 persistent worker 上完成同一 keep-alive
request/response 生命周期，均计入 framing、body 保留、逐字节验证、应用粒度 chunk 与连接退休。

advantage 为 HYLE/Beast op/s，越大越有利于 HYLE；每个顺序 15 个独立进程，置信水平 99.26%：

| workload / order | HYLE / Beast op/s | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| 128x1 KiB REST/stream / candidate-first | 62,138.71 / 61,837.99 | 1.00250 / [0.98076, 1.01702] | `PARITY` | `PASS` |
| 128x1 KiB REST/stream / peer-first | 62,083.94 / 63,012.44 | 0.98944 / [0.95496, 1.00659] | `PARITY` | `PASS` |
| 16x64 KiB fragmented / candidate-first | 17,834.58 / 5,881.35 | 3.03025 / [2.97461, 3.10162] | `WIN` | `PASS` |
| 16x64 KiB fragmented / peer-first | 17,655.12 / 5,829.72 | 3.06541 / [2.97468, 3.14026] | `WIN` | `PASS` |

A/A、5%/10% known-delta、timer floor、artifact/build/consumer/source/peer provenance 与采样前后稳定性全部
`PASS`；n=15 已裁决，没有触发 n=31。唯一有效 68-file bundle 位于
[`io-http-p1-v2-18bf62f8`](../../../../evidence/performance/io-http-p1-v2-18bf62f8/)；manifest、summary、
controls SHA-256 依次为
`757E15424213E1C0C036A0F8E4814E428BCB64CDB1648A7DBEECFD1A78C3A22B`、
`10B060A0014B78B6675133D650CFC5769265F14D9E2CF96AEDDF41E7545F6B3D`、
`B2187A8E78119A7BF63CB2BBBD6EE62AA5A1240AA0166D420832F42144CFDB86`。保留的 WSL EXT4
候选与 modular Boost peer 上严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、
`provenance=EXACT`、`environment_policy=CURRENT`。

这是 W1-WSL、GCC 14.2、Release/static、portable ISA 的 HTTP/1 工程 P1；它不声明 WAN、TLS、路由、
handler 调度、多 worker、完整 Nginx/推理服务器或 Formal SOTA，也不把 64 KiB 行的约 3x 优势外推到
模块整体。1 KiB 行诚实裁决为 parity，不因单次点值或执行顺序修改实现。

## 2026-09-09 `io_http2-v2` 当前工程证据（`PASS`）

不可变 candidate/tree 为 `4060e217c88d0ed35356b1ee21e5aefbe0c5b5d3` /
`cb91d5cba24706ddee390b51dab40fa12b2093ff`；peer 仍是 nghttp2
`85e300c79fb6dbcfa9c1013215c8710c1c2cd3d2`。本轮只移除 owner worker 普通 progress 的外层 session
monitor；公开 API、协议状态、body、窗口、chunk、ownership、validation、teardown、双顺序与固定 5%
margin 均与 v1 相同。双方仍在一个 persistent worker 上完成同一 h2c multiplex/flow-control consumer，
没有以轻语义、预分配或不同 body 形状获得结果。

advantage 为 HYLE/nghttp2 op/s，越大越有利于 HYLE；每个顺序 15 个独立进程，置信水平 99.26%：

| workload / order | HYLE / nghttp2 op/s | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| 64x1 KiB / candidate-first | 201,866.21 / 140,934.08 | 1.46578 / [1.38347, 1.50749] | `WIN` | `PASS` |
| 64x1 KiB / peer-first | 204,507.86 / 139,537.62 | 1.47803 / [1.40242, 1.52690] | `WIN` | `PASS` |
| 4x256 KiB / candidate-first | 5,341.92 / 5,313.12 | 1.01046 / [0.97785, 1.02041] | `PARITY` | `PASS` |
| 4x256 KiB / peer-first | 5,308.97 / 5,286.09 | 1.00024 / [0.96383, 1.02025] | `PARITY` | `PASS` |

A/A、5%/10% known-delta、timer floor、artifact/build/consumer/source/peer provenance 全部 `PASS`；n=15
已经裁决，没有触发预声明的 n=31 精度升级。第一次启动在任何采样前因 `/mnt/c` 不满足 W1-WSL EXT4
合同而被仪器拒绝，第二次仅由 shell fresh-path guard 退出；它们都没有样本或 bundle。唯一有效测量位于
[`io-http2-p1-v2-4060e217`](../../../../evidence/performance/io-http2-p1-v2-4060e217/)；manifest、summary、
controls SHA-256 依次为
`30172CD904CA90E641529DDEDAFE59C815EEEFAB5DE882F020CADF94656D394E`、
`44728C2EC8901B309F8FA37F798BADB9C9D24D3124E7F9AC498EF506FD810D94`、
`BE2643F97F369A1ED6C7DA87E9C2E6CF64310F98D5C1F966FE4BA1AFC65E589E`。
保留的 WSL live build/toolchain 上严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、
`provenance=EXACT`。这是 W1-WSL 工程 P1，不是 Formal 或全平台性能证书。

## 2026-09-09 `io_tls-v2` 当前工程证据（`PASS`）

不可变 candidate/tree 为 `18bf62f81a91975e0776439d88fca3891f10e560` /
`98632557e0237bac0bed3ca5521e59420d89176e`；peer 是 Asio
`8806a6803cde7054c3049d3666d3ec36786568c5`。本轮只修复 TLS ciphertext pump 在 write pressure 下的
全双工进展，并把 public `tls_write` 明确恢复为既有 `TransferResult` write-some 语义；合同、benchmark、
OpenSSL provider、verified localhost、persistent single-worker runtime、输入、teardown、双顺序和 5% margin
均与 v1 相同。没有通过缩短生命周期、移除验证、预建连接、改变消息数或把 blocking provider 与 async
runtime 混作 peer 获利。

advantage 为 HYLE/Asio op/s，越大越有利于 HYLE；每个顺序 15 个独立进程，置信水平 99.26%：

| workload / order | HYLE / Asio op/s | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| verified handshake / candidate-first | 1,120.78 / 1,026.02 | 1.09879 / [1.06150, 1.15334] | `WIN` | `PASS` |
| verified handshake / peer-first | 1,175.95 / 1,061.78 | 1.10161 / [1.07436, 1.13795] | `WIN` | `PASS` |
| 512x256 B encrypted stream / candidate-first | 214.79 / 207.07 | 1.04147 / [0.99905, 1.06048] | `INCONCLUSIVE` | `PASS` |
| 512x256 B encrypted stream / peer-first | 223.71 / 214.05 | 1.05265 / [1.00579, 1.06548] | `INCONCLUSIVE` | `PASS` |

握手双顺序均达到 5% practical margin 外的 `WIN`；small-record stream 两个点估计都偏向 HYLE，但置信区间
跨过 superiority 决策边界，因此只诚实声明 `INCONCLUSIVE` 与 5% non-inferiority `PASS`。A/A、5%/10%
known-delta、timer floor、artifact/build/consumer/source/peer provenance、host/input/artifact stability 全部
`PASS`；n=15 已裁决，没有触发 n=31。唯一有效 68-file bundle 位于
[`io-tls-p1-v2-18bf62f8`](../../../../evidence/performance/io-tls-p1-v2-18bf62f8/)；manifest、summary、controls
SHA-256 依次为
`D6B9A542FB091D3CFC221D1620F51A0FBAA63BD6FACF30C468EF0802232B5751`、
`71D09A4D8DAD0904A40C6AFE939A313EB04FE101864831C255549BE459E33C91`、
`15DB83BC0737C3DD0773D187264A33A5A57B859B0E7FB1AF35D4AB259EB2313C`。

保留的 WSL live build/toolchain 上严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、
`provenance=EXACT`、`environment_policy=CURRENT`，且没有 provenance renewal。这是 W1-WSL、GCC 14.2、
Release/static、portable ISA、同一 OpenSSL 的工程 P1；它不声明 Formal、WAN、跨机、所有 cipher/provider、
多 worker、所有记录大小或完整 HTTPS server 的普遍优势。

## 被替代证据

HTTP2/TLS/HTTP v1、旧聚合 IO benchmark、调参时间线和早期平台数表已由上面的五个当前 factor 证据取代。它们仍可从 Git 与原始 evidence 恢复，但不进入当前状态、不能拼接成跨候选结论。
