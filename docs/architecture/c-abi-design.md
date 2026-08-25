# Native FFI ABI 设计决策（C++ 库的跨语言动态接缝）

> **信源**：本文件规定 HYLE 对外 Native FFI ABI 的长期形状。C++ 产品设计见各模块
> architecture/guide，具体 ABI schema 见 `include/hyle/c/*.h`，逐入口绑定合同见
> `docs/guide/c-*.md`。施工步骤见
> [`c-abi-authoring-guide.md`](./c-abi-authoring-guide.md)。

## 0. 目标：让外部语言运行时直接加载动态库

HYLE 没有原生 C 客户。`include/hyle/c/*.h` 选择 C 语法，是因为 C calling convention 和
plain layout 是 Java、.NET、Rust、Python 都能描述的 **Native FFI ABI schema**，不是因为要
另做一套面向 C 程序员的产品 API。

目标消费者必须能直接加载 HYLE 的 `.dll` / `.so` / `.dylib`，解析公开导出并调用，不要求用户：

- 编译一段 C/C++ shim；
- 复制 HYLE 的 `static inline` 实现；
- 展开 C macro 或照着 native-C recipe 重写算法；
- 知道 C++ 类、STL、模板实例或异常 ABI。

ABI 形状必须允许下列常见绑定直接描述：

- Java：FFM API、JNA 或项目选定的等价 direct binding；
- .NET：P/Invoke；
- Rust：`extern "C"` / bindgen 生成的 `-sys` 层；
- Python：ctypes/cffi。

“C 头能被 C 编译器包含”只是 schema 卫生门，不是 FFI 产品验收。

本文件规定的是可供外部运行时绑定的 ABI 底座，不等于 HYLE 已发布 NuGet、JAR、crate 或 Python
package。当前 core 的 Java/.NET probe 证明 direct binding 可行并建立最低验收形状；正式 binding
package 仍需自己的高层 wrapper、版本选择、artifact 分发和所声明平台的 consumer matrix。

## 1. 三层必须分账

| 层 | 面向谁 | 允许的形状 | 能证明什么 |
|---|---|---|---|
| C++ source API | 原生 C++23 用户与 HYLE 内部模块 | 模板、RAII、concept、`std::span`、header-inline | C++ 用户任务可用，且源码组合自然 |
| native-C helper | HYLE 自己的桥接代码、可选 native shim/内核源码 | `static inline`、macro、透明 pointer primitive、C initializer | 只证明本地源码可编译、可内联；不属于 direct FFI |
| direct Native FFI ABI | Java/.NET/Rust/Python | 真实 exported symbol、opaque handle、固定布局 POD、callback+context、ptr+len | 无自编译 shim 即可从动态库取得运行时能力 |

三层可以共享实现，但不能互相冒充：

- C++ 有能力，不自动推出 FFI 有能力；
- C 头里存在函数名，不自动推出动态库有该符号；
- `static inline`、macro、transparent struct 和纯 C recipe 都不能计入 direct FFI 覆盖；
- 反过来，纯编译期能力没有合理运行时语义时，可以明确不进入 FFI，而不是为了“数量对齐”
  制造无价值导出。

Native FFI 事实以两项共同决定：公开 C schema 中带导出属性的声明，以及构建产物中 `hyle_*`
FFI namespace 的 export manifest；该 namespace 内多出或缺失都属缺陷。同一 shared library 还可
合法承载 C++ dynamic SDK 的 decorated exports，它们由 C++ public API/build contract 另行审计，
不计入 FFI manifest。行为型 inline/macro helper 原则上移出 schema 头；迁移期仍共处时也必须
单列为第二层，不能混入导出清单或 FFI 能力矩阵。ABI tag/flag 常量等 declarative macro 只是
schema metadata，binding 生成后保存其值，仍不是动态能力。

## 2. 先按跨语言任务裁决，而不是翻译 C++ 符号

| C++ 能力形状 | Native FFI 裁决 | 标准形状 |
|---|---|---|
| 有真实跨语言价值的无状态 runtime 函数 | 直接导出 | `hyle_*` 自由函数 |
| RAII / owning resource | 导出资源能力 | opaque handle + create/open + close/destroy |
| 模板算法或容器 | 先确认跨语言任务，再类型擦除 | runtime tag/size/stride + opaque state；不能只给 typed macro |
| callback / async completion | 导出时显式建模 | function pointer + context + register/cancel/drain 合同 |
| 字符串、数组、序列化结果 | 明确数据方向和所有权 | ptr+len、caller buffer-fill，或 library-owned buffer + 对称 free |
| 高频逐元素操作 | 不能照搬细粒度 C++ API | batch-first/fused export，摊薄 P/Invoke/FFM/ctypes 边界成本 |
| compiler hint、`assume`、`restrict`、unroll | 不进入 direct FFI | C++/native-C source helper；托管调用时没有可保持的 TU 合同 |
| 编译期 layout/alignment 常量 | 通常不进入 direct FFI | 只影响本地编译布局；运行时硬件查询归拥有该事实的模块 |
| 单地址 prefetch | 不导出 | FFI hop 通常比 hint 更贵；需要时融合进处理整批数据的 runtime op |
| transparent intrusive hook/list | 不作为 managed FFI 容器 | stable native address 和手工 lifetime 不适合 GC；有队列场景时设计 opaque、批量的上层资源 |

判据是“Java/.NET/Rust/Python 能否直接完成被承诺的运行时任务”，不是“C++ 每个函数是否有一个
C 拼法”。如果答案依赖调用方编译 recipe，它属于 native-C helper；如果该能力本来只在编译期
影响代码生成，则明确记录不导出即可。

## 3. ABI 基本积木

### 3.1 Exported symbol

所有 direct FFI 入口都是 unmangled `extern "C"` 自由函数，并使用模块三态导出宏。共享库 export
manifest 必须锁住确切名称；不能把链接器偶然保留的 inline/template 实例当产品 ABI。静态库可供
C++/嵌入式交付，但不能替代动态直绑证据。calling convention 也是 ABI：支持平台统一使用并
记录 HYLE 的 C convention，构建参数不得静默改成另一 convention；.NET 等 binding 必须显式匹配。

### 3.2 Opaque handle 与 managed lifetime

跨边界资源用不透明指针或明确宽度的 token 表示，禁止暴露 C++ 对象布局。每个 owning handle
必须回答：

- 谁创建、谁销毁，NULL 是否可传；
- destroy/close 是否幂等，失败后 out 参数是什么；
- borrowed child、view、错误字符串和 buffer 有效到何时；
- callback/异步操作仍在飞时，cancel、drain 与 destroy 的顺序；
- 哪个线程可以使用或释放。

绑定层必须能自然包成 Java `AutoCloseable`、.NET `SafeHandle`、Rust `Drop`、Python context
manager/finalizer。finalizer 只能兜底，正常路径仍需确定性 close；库不得依赖 GC 恰好及时运行。

### 3.3 POD 与演进

跨 ABI POD 只含固定宽度整数、浮点、指针、显式长度和其他已裁决 POD。tag/flag 不直接使用宽度
不稳定的 C enum。可扩展输入以 `struct_size` + `reserved` 开头，桥接层按可读完整字段解码；
小而冻结的 snapshot 才适合值返回。C/C++ 两面用 standard-layout、trivially-copyable、size、
alignment、offset 和 tag value 的 `static_assert` 锁定。

POD “C 可写”不等于“托管语言好绑”。嵌套自引用指针、平台相关 bit-field、柔性数组、未说明
packing 或需要调用方维护内部不变量的透明结构，都不应进入 direct FFI。

### 3.4 Callback

callback 一律是 plain function pointer + `void* context`，并明确：

- 同步调用还是可能在返回后保存；
- 调用线程、是否并发、是否允许重入；
- context 和 delegate 必须存活到哪个 unregister/drain 点；
- callback 返回值如何停止、取消或报告错误；
- native 侧是否会在 destroy 后再次回调。

Java/.NET smoke 必须真实保活 callback/delegate、触发 GC 压力并完成 unregister/drain，不能让
一个 C callback test 代替 managed runtime 的 pinning 与线程行为。
Java upcall 若可能被 native worker thread 调用，stub 必须由可跨线程访问且覆盖完整注册期的
shared arena 承载；confined arena 只适合明确限定在 owner thread/同步调用栈内的 callback。
Java `Throwable`、.NET exception 或其他 managed unwind 绝不能穿过 native frame：binding trampoline
必须在 managed 边界内捕获，并按已设计的 status/stop/record 通路报告。

### 3.5 Buffer 与字符串所有权

每个 pointer 都同时写清 direction、element type、element count/byte length、NULL 与 zero-length
规则、alignment、借用/拥有关系和有效期。优先级是：

1. 调用方提供 buffer，函数返回 required/written size；
2. 库返回 borrowed view，并给出精确失效点；
3. 库分配 owning buffer，同时导出同模块的对称 free。

禁止要求 Java/.NET/Python 用本语言 allocator 释放 HYLE 内存，也禁止返回没有长度或生命周期说明
的裸字符串。错误消息同样遵守这套规则。

### 3.6 Batch-first

跨 managed/native 边界的调用成本可能远高于一次容器操作、prefetch 或小算术。公开运行时能力应
优先接受数组、span-like ptr+count、descriptor batch 或 command list，并明确 partial completion
和 per-item error。逐元素 primitive 可以作为 C++/native-C 内核实现，但不能因为它“inline 很快”
就声称 FFI 路径也快。

### 3.7 Shared library 是所有 FFI 对象的根生命周期

函数地址、opaque handle、borrowed pointer、callback/upcall stub 与仍在执行的 native worker 都依赖
其所属 shared library 保持加载。生产 binding 通常把 HYLE libraries 固定到进程生命周期；若确实
支持显式 unload，顺序必须是：停止新调用，cancel/drain 异步工作，unregister callbacks，关闭全部
handles 并复制/放弃 borrowed views，释放 managed callback arenas/delegates，然后按实际依赖图逆序
卸载模块；若该 package 加载了 `hyle_core`，它在所有依赖者之后卸载。卸载后的函数指针、POD
内嵌指针和错误字符串全部失效。

## 4. 类型擦除的正确用途

类型擦除服务于真实 runtime 能力，而不是把 template 名字机械改成 `void*`：

| C++ | 合理的 FFI 形状 |
|---|---|
| `GrowablePool<T>` | 若跨语言确有池化场景：opaque pool + slot size/alignment + batch acquire/release |
| `Allocator::construct<T>` | 只导出 raw allocation/buffer ownership；对象构造由绑定语言负责 |
| `Tensor<T, N>` | runtime dtype/rank/shape/stride descriptor + opaque storage |
| `std::function` consumer | callback pointer + context + 明确保存/取消/线程合同 |

若类型擦除后仍要求调用方嵌入 native hook、执行宏或维护 C++ 不变量，它没有成为 direct FFI。

## 5. 错误跨边界

- C++ 异常绝不跨 `extern "C"`；桥接层捕获允许出现的外来异常并翻译。
- 可恢复失败使用统一 `hyle_status` + out 参数，out 进入时清空，成功才转移所有权。
- owning status/message 的销毁和借用期限必须适合 managed wrapper；不得依赖 TLS last-error。
- 热路若用 NULL/计数表达正常耗尽，必须与非法输入和系统失败无歧义；批接口需说明 partial
  completion。
- 生命周期、foreign/double free、违反并发合同等程序错误仍 fail-fast，但文档必须让绑定层可以
  避免这些状态。

## 6. 版本与动态发现

Native FFI 继续采用普通 `extern "C"` 导出，不恢复全模块函数指针表或 `api_get`。但 separately
distributed managed bindings 与 native runtime 确实会独立演进，因此使用**全局版本、逐 artifact
握手**：

- high 16 bits 是 ABI major，low 16 bits 是 minor；
- core artifact 的 bootstrap 是 `hyle_ffi_abi_version`；它只认证被加载的 `hyle_core`，不能替另一
  个模块 DLL/so/dylib 背书；
- 以后某模块经真实 runtime 任务裁决为需要 direct FFI 时，其自身 shared artifact 必须导出
  bootstrap `hyle_<module>_ffi_abi_version`，返回同一 `HYLE_FFI_ABI_VERSION`；FFI=`N/A` 的模块
  不为凑形式增加 bootstrap，也不因此依赖 core；
- binding 从待调用模块的 exact library handle 先解析该 artifact 的 bootstrap；
- major 必须相同，runtime minor 必须大于等于 binding 生成时所需 minor；
- 通过后才解析该 binding 需要的其余 symbols；
- ABI version 与 product `major.minor.patch` 分开，后者只描述 HYLE release identity。

这不是函数表：符号仍由 loader 直接解析，minor 增长只允许增加 exports，或扩展本来就采用
`struct_size` 协议的 optional 尾字段。**当前没有任何已发布 binding 或外部兼容对象**：首个 FFI
产品发布前，现有 symbol、版本值和 bootstrap 方案都可以直接修改或删除，不保留无人消费的历史
别名。某个 binding 一旦正式发布，它实际依赖的 bootstrap 名、C calling convention 与精确
`uint32_t(void)` 签名才跨后续 major 保留，否则旧 binding 连安全拒绝都做不到；扩展协议时增加
新名字，旧入口只为已发布消费者保留。模块 bootstrap 不是模块私有版本号，只是从该 artifact
查询全局 ABI generation。

一个 binding 同时调用多个 HYLE shared artifacts 时，必须逐个握手，并从同一个 package/install
root 装载完整依赖闭包；core 握手通过不能证明另一 artifact 来自同一套产物。官方 binding/package
还必须把单一 library root、依赖 staging 与禁止混用不同安装根作为原子发布合同。该门归
build/package 复审；在它完成前，非 core 模块不能借 core 的绿色证书取得 direct FFI 证书。符号
可发现性仍由各 artifact 的 FFI-subset export manifest 和真实动态加载 smoke 证明。

## 7. 证据：C smoke 不能替代 FFI smoke

每个承诺 direct FFI 的模块至少需要：

1. C schema self-contained 编译和 ABI layout asserts；
2. shared artifact 中 FFI namespace 的精确 export manifest；C++ decorated exports 单独验收；
3. Java 从产物目录直接 load，先验证该 exact artifact 的 FFI ABI version，再调用该模块真实 exports；
4. .NET P/Invoke 对同一 shared artifact 做相同的逐 artifact ABI 握手与真实 exports 路径；
5. 按实际 ABI shape 覆盖 POD、handle、buffer、callback、error 与 unload lifetime；不存在的 shape
   明确记 `N/A + 理由`，不得为凑矩阵制造导出；
6. 若存在数据面，运行 batch 正常 workload，确认没有把逐项 FFI hop 当正常路径；
7. Windows/Linux/macOS 对各自 shared artifact 的 loader、calling convention、位宽、导出名和
   最小动态调用真跑；模块收口至少有一台宿主完成 Java + .NET 双 managed smoke。

这些测试不得编译用户侧 C shim。纯 C 测试仍可验证 schema 和 native-C helper，但只能记在第二层。
Java/.NET 的真实运行结果是 managed lifetime 与 marshalling 的最低硬门。Rust/Python 在 HYLE
正式承诺对应 binding package 时进入该 binding 的发布矩阵；不能为了凑语言数量，用四份只调
version 的重复 smoke 代替一个覆盖真实 handle/buffer/callback 的托管消费者。
一旦发布官方 Java/.NET binding package，该 package 支持的每个平台都必须运行自身 consumer；
模块施工期的一台宿主双 smoke 不能冒充 binding package 的发布矩阵。

## 8. core 是边界裁决示例

core 的 direct FFI 只有 runtime ABI/product identity 是合理的：

- compiler contracts、cache layout 常量都是 translation-unit/编译期能力；
- 单地址 prefetch 跨 FFI 调用得不偿失，需要时应融合到所属模块的 batch runtime op；
- transparent intrusive list 依赖稳定 native 地址、嵌入 hook 和手工脱链，不是 managed 容器 ABI；
- runtime cache-line observation 属 platform，不因 core 有 cache layout 常量就复制一份导出。

因此 core 只导出 `hyle_ffi_abi_version`、`hyle_version`、`hyle_version_string` 不是“遗漏”。
旧 native-C prefetch/list/compiler helper 既不能计作 Java/.NET/Rust/Python 覆盖，也不应继续混在
FFI schema 中；它们已从 core schema 移出。这个区分是所有模块复审的模板。

## 9. 落地纪律

- HYLE 主体保持 C++23；Native FFI schema 使用语言中立 C 声明，但没有原生 C 产品旅程。
- 每个 schema item 必须标为 direct export、ABI POD/callback，或 native-C helper；禁止模糊的
  “C API 已覆盖”。
- bridge 编进能力所属模块，显式构造 C++ 对象，不 reinterpret C++ 布局。
- create/destroy、buffer ownership、callback lifetime 和 batch semantics 是 API 设计的一部分，
  不是绑定文档补丁。
- 编译期/source-only 能力可明确不导出；runtime 跨语言任务不能用 inline/macro/recipe 免责。
- 关闭一个模块前同时核对 C++ source、native-C helper 和 direct FFI 三份矩阵，并以真实
  Java/.NET 动态加载结果签署 FFI 门。
