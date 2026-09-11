# Mem v8：MemoryResource 类型可见性

## Closure card

- component / owner / kind：`mem` / Mem / `REOPEN`；round=`mem-v8-resource-type-visibility`；
- baseline：`5eaf7af55f9844e3175edbd958507eb60fb1208c`；前一 terminal record 是
  [`mem-v7-logical-range-suballocator`](mem-v7-logical-range-suballocator.md)；
- trigger 与证据：build v13 为 shared factor 启用正确的 default-hidden 后，MacBook Air arm64 的真实
  Sched source consumer 在链接 `hyle_mem_vm` 时找不到 `typeinfo for hyle::mem::MemoryResource`。
  `MemoryResource` 是允许其它 HYLE factor/用户派生的公开多态 provider base，但当前只给 out-of-line
  virtual 方法标了 `HYLE_MEM_API`，没有给类型/vtable/RTTI 建立 shared 边界；
- 正常 consumer：另一个 shared factor 或同工具链 C++ 用户实现 `class Provider : public
  MemoryResource`，通过 `MemoryResourceRef` 交给 Buffer/VM 上层；
- 主/反偏置纵切：Mem-vm 的 `BorrowedMappedResource` 跨 DSO 派生与 source/static/shared Mem consumer；
  无 RTTI 使用的普通 allocator/buffer 路线不改变；
- scope：只给公开 `MemoryResource` 类添加现有 `HYLE_MEM_API` 类型标注，删除冗余成员级重复标注需有
  编译器证据才允许；
- non-goals：改变资源虚表、方法、layout、ownership、分配算法、异常/RTTI policy、Native FFI 或性能；
- 直接依赖/消费者：依赖不变；直接验证 `mem_vm`，上层 Sched 继承其交付回放；
- 性能：`P0`，visibility metadata 不进入调用路径，vtable 与对象 layout 不变；
- 预算/验证：一次声明修正；Windows/Linux/macOS shared link，Mem correctness，Mem 与 Sched selected
  source/static/shared、实际 export hygiene、文档/closure；不跑 P1；
- stop：若类级标注改变 layout、要求新增 trampoline 或导致静态/source 语义分叉，则停止并回到审计。

## 有限改动集

| ID | 问题 | 修复 | 验收 |
|---|---|---|---|
| M8-01 | public polymorphic type 的 RTTI/vtable 被 default-hidden | `class HYLE_MEM_API MemoryResource` | 三平台跨 factor 派生/链接；ABI size/alignment 不变 |

## AUDITED 裁决

全量公开头/实现扫描确认 `MemoryResource` 是 Mem 唯一允许用户和其它 factor 跨 binary 派生的公开
多态 extension base；`PmrResource` 只在调用方 TU 内派生标准 PMR，其他 pool/arena 类型不承担跨 DSO
RTTI/vtable 身份。类级 default visibility 是该现有能力的必要交付语义，不是新增 API。

有限改动冻结为：把 `HYLE_MEM_API` 移到 `MemoryResource` 类声明，并删除该类 out-of-line virtual 上
重复的成员级标注；方法集合、顺序、签名与定义不动。该属性不占对象 storage、不增加 vtable slot，也
不进入指令路径。当前进入 `AUDITED`，下一提交才允许修改公开头。

## Candidate

- immutable product/build candidate：`ad5bff895125fb0d95281b36559010435c3d7892`；
- `MemoryResource` 改为类级 `HYLE_MEM_API`，冗余成员级标注已删除；公开方法、虚表顺序、对象
  layout、资源算法和常态成本不变；
- Windows Mem/Mem-vm correctness 与 Sched correctness 通过；Windows、Linux GCC 14、MacBook Air
  arm64 的 Sched source/static/shared relocated consumer 均通过；
- MacBook Air 上原先的 `typeinfo for hyle::mem::MemoryResource` 跨 factor 链接失败已消失；
- candidate 后只允许补 terminal 证据、文档与状态，不再修改产品实现。

## 验证

| Gate | 适用风险与命令/证据 | Verdict |
|---|---|---|
| scope | candidate 相对 baseline 只改 `resource.h` 的类型可见性；方法集合、实现和依赖不变 | PASS |
| architecture | `MemoryResource` 是 Mem 唯一公开、可跨 factor 派生的多态 extension base；类型级 export 与其真实语义一致 | PASS |
| correctness | Windows Mem/Mem-vm correctness 全过；方法、虚表与 allocator 路径未改 | PASS |
| delivery | Windows、Linux GCC 14、MacBook Air arm64 的 Sched source/static/shared relocated consumer 全过 | PASS |
| usability_docs | 用户仍使用同一个 `MemoryResource`/`MemoryResourceRef` API；不新增模式、开关或迁移负担 | PASS |
| red_team | 全量 Mem public type/派生扫描没有发现第二个缺标的跨 DSO extension base；Mac 原始失败由同一回归闭合 | PASS |
| engineering_performance | P0：visibility 不占对象字段、不增加虚表项或指令；v7 的 P1 运行时证据因算法字节未改而继续有效 | PASS |

## 红队与裁决

- 反证：在 default-hidden Mach-O 上用真实 `BorrowedMappedResource` 派生链，而非人为 RTTI probe；原失败
  能稳定复现，类级 export 后 source/static/shared 同时通过。Windows/ELF/Mach-O artifact 扫描也没有
  因修复产生 plain-C 或 vendor 导出；
- 未决限制：共享 C++ SDK 仍要求兼容工具链/运行库；这不是稳定 Native FFI；
- reopen triggers：新的公开跨 factor 派生类型出现同类链接失败；`MemoryResource` 的 API/layout/虚表或
  调用成本变化；真实 consumer 发现资源生命周期缺陷；
- LOC（candidate）：全库 560,631；自实现不含测试 100,233；Mem 实现 8,263；
- 最终状态与理由：`CLOSED`。唯一可复现 shared 类型身份缺陷已在三平台真实下游闭包中消失，且没有
  改变 Mem 的功能、形状、算法或性能合同。
