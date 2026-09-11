# Sched v11：私有 fcontext 共享库边界

## Closure card

- component / owner / kind：`sched` / Sched / `REOPEN`；round=`sched-v11-private-fcontext-export`；
- baseline：`526dd5aee87b33936835fbb7cd9c447f7574d489`；前一 terminal record 是
  [`sched-v10-execution-priority`](sched-v10-execution-priority.md)；
- trigger 与证据：对已安装 Windows shared SDK 使用 PE export table 扫描，确认
  `hyle_sched.dll` 意外公开 `jump_fcontext`、`make_fcontext`、`ontop_fcontext`。它们是 Sched 内部
  Boost.Context 汇编调用约定，不是用户 API，却违反 `Native FFI=N/A` 和私有实现边界；
- 正常 consumer：C++23 用户只链接 `hyle::sched`，经公开 Scheduler/Fiber/Task API 使用 source、static
  或 shared 交付；不直接解析或调用 fcontext 符号；
- 主产品纵切：Windows x64 shared Sched consumer 创建、执行和关闭 scheduler/fiber，同时 artifact 扫描
  证明 raw 汇编 procedure 不在 PE export directory；
- 正交反偏置纵切：source/static 交付继续链接相同汇编 object；Unix ELF/Mach-O 既有 `.hidden`/
  `.private_extern` 语义由 shared smoke 复验；
- 跨模块 public journey：公开 Sched API → Sched binary 内 fiber switch → private fcontext object；IO 等
  上层消费者只经过公开 C++ API；
- scope：仅把 Windows x64 MASM procedure 从 DLL `EXPORT` 改为 object 内 `PUBLIC`，并继承 build v13
  artifact 检查；
- non-goals：改变 Sched API、调度语义、fcontext 指令、对象 layout、调用约定、第三方版本、性能仪器
  或 Native FFI；
- 直接依赖 / 直接消费者：依赖图不变；直接消费者 IO 不需迁移；
- 性能 tier 与原因：`P0`，汇编指令与调用边完全不变，只删除 PE export-directory metadata；
- 调查与纠正预算：一次 MASM directive/object 检查、一个产品修复、一次三交付；
- 预声明验证：MASM object symbol/directive 反证；Sched correctness；Windows source/static/shared relocated
  consumer；shared export checker；适用 Unix selected smoke；构建/文档/closure 检查；
- stop / blocker：若 `PUBLIC` 无法让同 DLL 内 C++ object 解析 procedure，或汇编/codegen 发生变化，
  立即撤回并回到 `AUDITED`；不为规避检查器包装第二套 trampoline。

## 能力与形状审计

公开 Sched 能力、语义和 consumer API 完全继承 v10；本轮没有新增能力。raw fcontext 是 compiled
implementation seam，不是原子产品能力，也不应由 public header 或 symbol table形成旁路入口。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| S11-01 | Windows MASM `PROC EXPORT` 写入 PE export directive | `BOOST_CONTEXT_EXPORT=PUBLIC` | 同 object 中符号仍 external，且无 `/EXPORT` directive；三交付通过 | 已完成 |

### API / usability freeze

公开 C++ API、ownership/lifetime/failure/concurrency、默认值、allocation/atomic/lock/syscall 成本全部不变。
普通用户仍只链接 `hyle::sched`；不能调用或依赖三个 raw procedure。无需 P1。

## 七问

1. fcontext vendored 路径、许可证和版本不变；本轮不升级第三方。
2. Sched 能力和真实 consumer 不变；修的是未承诺的导出旁路。
3. 内部依赖和下层加速设施不变。
4. 调用指令与复杂度不变；以 object/codegen P0 反证，不计时。
5. 删除意外 plain-C exports 后才满足 `Native FFI=N/A`。
6. source/static/shared 均需继续链接；shared 额外验证 export table。
7. C++ 用户入口不变；分发文档明确 shared SDK 不是 FFI。

## AUDITED 裁决

Windows x64 MASM 的 `EXPORT` 会把 procedure 写入 object 的 linker directive；`PUBLIC` 只让同一 DLL
内的 C++ object 可解析符号。对同一 vendored `jump_x86_64_ms_pe_masm.asm` 的独立编译反证显示，
`PUBLIC` object 保留 external `jump_fcontext`，且不再产生 `/EXPORT:jump_fcontext` directive。ELF 源码
已对三项入口声明 `.hidden`，Mach-O 源码已声明 `.private_extern`，没有第二处产品算法缺口。

有限改动集冻结为唯一 S11-01；状态进入 `AUDITED`，下一提交才允许修改产品 build seam。公开 API、
汇编指令、调用约定、模块依赖和性能 workload 均不改变。

## Candidate

- immutable product/build candidate：`e07c025ee7f97e7f132b2daf42abaa5ac23bfb63`；
- `BOOST_CONTEXT_EXPORT` 已从 MASM `EXPORT` 改为 `PUBLIC`；fcontext 三个汇编源文件和 C++ 调用方没有
  字节改动，公开 API、layout、算法、指令路径和常态成本不变；
- build v13 同时令 compiled shared factor 默认 hidden，并让显式 `HYLE_*_API` 保持 public；
- Windows MSVC Release 的 Sched correctness 通过；selected Sched source/static/shared relocated consumer
  全部通过；shared SDK 实际扫描 8 个 HYLE DLL、492 个导出，未发现 raw fcontext、`mi_*` 或 plain
  `hyle_*` C 符号；
- candidate 后只允许补跨平台交付、红队、文档和 terminal 状态证据。

MacBook Air 候选后交付首次运行在链接下层 `mem_vm` 时发现 `MemoryResource` RTTI 未导出。这不是
fcontext 修复或 Sched 算法缺陷，但 candidate 的完整依赖闭包尚不能在 default-hidden Mach-O 下交付，
因此依法由 `CANDIDATE` 退回 `AUDITED`。修复由 Mem v8 与 Mem-vm v3 独立卡拥有；完成后 Sched 需基于
新依赖提交重新签 candidate，不得沿用 `e07c025e` 的跨平台结论。

### 完整依赖闭包重新签发

- immutable product/build candidate：`ad5bff895125fb0d95281b36559010435c3d7892`；
- Sched 的 fcontext 修复仍是 `e07c025e` 中唯一产品 seam 变化；新候选只纳入已依法修复的 Mem 类型
  可见性依赖；
- Windows、Linux GCC 14、MacBook Air arm64 的 Sched source/static/shared relocated consumer 全部
  通过；Windows Sched correctness 通过；
- shared artifact 实扫分别覆盖 Windows 8 DLL/492 exports、Linux 8 ELF/535 exports、macOS 8
  dylib/533 exports，均未发现 raw fcontext、`mi_*` 或旧 plain `hyle_*` C 入口；
- candidate 后只允许补全量 package、红队、文档和 terminal 状态证据。
