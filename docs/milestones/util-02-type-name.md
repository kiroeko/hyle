# util-02：编译期 type_name + 类型哈希

| 字段 | 值 |
|---|---|
| **阶段** | P3（util 簇）|
| **模块** | util/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；util 14→23 case）|

> **cloc**：**生产 44 行**（type_name.h，header-only constexpr）+ **测试 50 行**。
> **clay 对比 146 → HYLE 44（不到 1/3）**：clay 硬编码"编译器(MSVC/Clang/GCC) × C++标准(17/旧) × CUDA"的签名串矩阵(6+ 个写死的 prefix/suffix);HYLE 用**一个 probe 派生**所有偏移——更短**且更 robust**(见下)。9 测试(含 constexpr static_assert + 双编译器切片验证)证明非 stub。

## 核心改进：派生偏移 vs clay 硬编码（对抗自审驱动）

clay 把每个编译器的**完整 pretty-function 签名串**写死作 prefix/suffix 去 strip——一旦重命名函数/命名空间,切片静默错位(返回垃圾)。HYLE 用 magic_enum/ctti 的 **probe 技术**:`wrapped_type_name<double>()` 拿到当前编译器的签名,`find("double")` 定位替换点,**算出** prefix/suffix 长度,对任意 T 通用。重构签名不会破。

两平台实测切片正确:MSVC `__FUNCSIG__`、GCC `__PRETTY_FUNCTION__` 格式迥异,同一份 probe 代码都切对——证明 robust。

## clay API diff + 决策

- **type_name<T>()**:覆盖 clay fully_qualified_name/type_name。返回 `std::string_view`(借用编译器静态串,视图非拥有,conventions-approved)——**不返回 string_ref**,保持 util 不依赖 container(与 string_ref 归属待决一致,不提前耦合)。
- **type_hash<T>()**:覆盖 clay static_murmur_hash。用 **FNV-1a**(constexpr)非 murmur——XXH3(HYLE 运行期哈希)非 constexpr,murmur 编译期版未移植;FNV-1a 等效用途(编译期稳定类型 ID)且 trivially constexpr。coverage-map murmur 行据此更新:编译期 type 哈希 = FNV-1a(util-02)。

## 对抗式自审（六类）

- **① 正确性**:`double` 作 probe 哨兵——它不出现在 `hyle::util::detail::wrapped_type_name`/`std::string_view` 拼写里,故签名中唯一,find 定位准;prefix/suffix 对所有 T 通用(签名仅 T 处不同)。unknown 编译器 wrapped 返回空→kSupported=false→type_name 返回空(安全降级,不 UB)。
- **② 性能**:全 constexpr,编译期完成,运行期零开销(static_assert 测验证)。
- **③ 规范**:返回 string_view(视图);header-only;不依赖 container。符合 §8。
- **④ clay 覆盖**:type_name + type_hash 全覆盖;murmur→FNV-1a 已登记。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:比 clay 的宏分支矩阵简洁得多(44 vs 146),probe 一招制敌。
- **已知局限**:名字是编译器拼写(MSVC 加 "class"/"struct" 前缀),**不跨编译器字符串相等**——只用于单编译器构建内的稳定 ID + 诊断,测试用 substring 容错。已注释。

## 交付物

- `include/hyle/util/type_name.h`（type_name<T> + type_hash<T>，constexpr）
- `tests/util/type_name_test.cc`（9）

## 参考

- clay `utils/type_name.h`(146)。conventions §8（编译期 type_name + 哈希,集中维护 + 单测）。
- magic_enum / ctti 的 probe-derived 偏移技术。
