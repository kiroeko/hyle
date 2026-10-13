# G01 CPU 精确页驻留读取有限轮终审

状态：CLOSED，2026-10-14；provider `compute_provider_ggml` candidate
`1e9cdfd45307ea4a15b92374377ad2b268203193`，runtime `compute_ggml_runtime` candidate
`42f98b94f33bdbb252b9259ece33533cf763fa41`。授权、baseline与有限CP-01/02/03/CP-A01见
[施工卡](inference-g01-cpu-paged-resident.md)。仅签已验CPU page16域，整个G01/G02不代签。

## 七问与七门

1. 第三方仍原ggml CPU-only ffa4与mixed456/IQ1锁；两个原CPU flash只改有效长度与
   四个K/V读点的地址映射。许可、patch/tree hash认证，dot/softmax/SIMD/reduction不复制。
2. 真实两请求GQA8/2、F16/BF16、D64/128/256、page16、causal/noncausal、反向/共享页与tail；
   独立非零double、64步generation/topology复用＋64步真实增长/跨页/结束/重入/重排、
   双topology槽/旧output、未引用页可写与共享页不可写、预算/逐分配失败/已接纳错误退休通过。
3. 复用原BatchProgram/Invocation、F1 exact page frames、CPU Queue/Completion/native slots；
   冷owned layout保页表与完整topology，复用原Recipe，不新增普通Request/Program页字段。
4. warm数学直接读原physical roots及私有owned页表，无KV历史复制或每Q/head物化。
   page16用等价shift/mask，generic原地址式保；query/output/causal mask/private metadata计费，
   private池总上界UNKNOWN。下列P1只签不可变ready历史，不签producer/growth生产的性能。
5. 原public Batch入口与CPU工厂；private PagedRead/原vendor地址参数，不引入public vendor
   类型、FFI、兼容层或另一executor/queue。普通非paged数学/资源/退休合同未变。
6. Windows最终源五tests；WSL ASan+LSan/UBSan/TSan各14tests；relocated static3/shared3
   各三个public paged/Direct+Graph ragged/resident消费者；实际Air Mac16,12/M4/arm64/
   macOS27.0.1/AppleClang21的CPU+Apple shared原mixed456 CPU14tests全部PASS。
   见[最终风险](../../../../evidence/correctness/inference-g01-cpu-page16-risk/README.md)和
   [三交付与mixed核](../../../../evidence/correctness/inference-g01-cpu-page16-delivery/README.md)。
   公共API/头未新增，原factory逐头证据保；不把宿主san或Air CPU签成GPU资格。
7. [Compute guide](../../../user/compute.md)给普通CPU工厂→Batch prepare/update→exact arena
   read→run→output的形状、旧结果阻止复用与支持域；cold topology变动重建、private/capture/
   GPU与联合分页限制可发现。source/static/shared沿同一public target，不要求internal target。

scope、architecture、correctness、delivery、usability_docs、red_team、engineering_performance
在本有限域均PASS。终审复核actual roots/页table origin/逻辑有效长度、owned借用、两个原核、
旧结果/失败退休、对照冷cache收费、consumer bytehash与不可变SDK身份；未发现未决本域缺陷。
原selector共享bench unknown_paths和归属问题保原BLOCKED计划/独立工具卡，不扩大到全库。

## 定向性能与停止

[最终P1](../../../../evidence/performance/inference-g01-cpu-page16-w1-p1/README.md)：clean source
`f9a689be320a9330b507a274918090ce44a17813`，contract
`3d05ed563439cb0f376c2f3ed1de2ac9c1b2f7e1eb5e9285a81f071e3badd574`，实际W1-WIN/
i9-13900K/MSVC19.50/logicalCPU2/单native线程。16A/A、120产品/30 controls、n15按原规则
停止，1657.156s；QUALIFIED、三个consumer-evidence/全部controls/5%非劣PASS、verify0。

| 场景 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| BF16 Q1/17 KV1024/768 D256 causal | 0.9867–0.9912 | PASS / PARITY | LOSS |
| BF16 Q64/128同KV D256 causal | 0.9882–0.9932 | PASS / PARITY | LOSS |
| BF16 Q1/17 KV31/47 D64 noncausal | 0.9869–1.0084 | PASS / PARITY | INCONCLUSIVE |

两点D256约0.7–1.3%稳定LOSS保留，不由5%门改写为方向parity。peer为独立pristine
ffa CPU AVX2 DLL，actual compute image/ABI核验；两arm同Batch/metadata/F1权限/Queue/
Completion/output边界。peer只消费不可变ready历史，冷按valid行另建contiguous cache，
额外D2563670016bytes/D6439936bytes明确收费、原page owners全程保活；没有warm复制。
原[1695.110s诊断轮](../../../../evidence/performance/inference-g01-cpu-paged-w1-p1/README.md)
因provenance hash旧值为DIAGNOSTIC，D64原约18%地址成本症状保留，不重标资格。两正式
活动累计3352.266s；原配置/fixture/ticket/SDK/link/mtime与patch身份FAIL均在施工卡/原证据保。

有限CPU page轮CLOSED；停止重复该已签域。reopen仅实际已验域数学、address/root/owned
layout、生命周期/错误退休、allocator/资源费用缺陷，或新真实消费机制。GPU原生高效分页、
G02联合exact guards/native绑定、G03producer、G06低成本规划、G10/private预算与其它G项仍
必须完成；不签任意page/shape、producer性能、general capture或完整模型token/s。

LOC（原loc-report/cloc2.08，5ce工作树测量）：全库2892818、自实现不含测试175246、
CPU provider2448、runtime自实现402；vendor patches、测试/benchmark均不冒充自实现模块LOC。
