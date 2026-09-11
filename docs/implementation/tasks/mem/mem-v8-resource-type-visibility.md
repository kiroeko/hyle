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
