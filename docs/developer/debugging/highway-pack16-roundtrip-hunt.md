# Highway Pack16<11>/<13> 满位宽 round-trip Hunt(vendor 算法 bug —— HYLE 对拍揪出、独立探针坐实、已绕过)

> 调查日志(已结案)。触发:simd-09-bitpack 薄封装 Highway bit_pack 时,round-trip 对拍(pack→unpack 必须还原)在 u16 的 bits=11/13 失败。
> 关联:[`bitpack_body.inc`](../../../src/simd/bitpack/bitpack_body.inc)(绕过点 `bitpack_width_borrowable`)、[`bitpack.h`](../../../include/hyle/simd/bitpack.h)。上游:`third_party/highway/hwy/contrib/bit_pack/bit_pack-inl.h` 的 `Pack16<11>`/`Pack16<13>` 手写特化。
> 结论先行:**Highway 的 `Pack16<11>` 和 `Pack16<13>` 手写特化不能对满位宽(全 11/13 位)值 round-trip**(独立探针实测 14/256、102/256 lane 不还原)。根因:Highway 自己的 `bit_pack_test.cc:55` 用 `Random32(&rng) & kBits`——把输入和**字面量位宽数**(11、13)按位与,而非 `& ((1<<kBits)-1)` mask,所以测试永远只喂 ≤kBits 数值本身那几个小值,满位宽路径从未被测,bug 未测上游就发货。其余所有宽度(Pack8 全 1..8、Pack32 全 1..32 泛型 unroller、Pack16 另外 14 个宽度)对拍全净。**处置**:对 (u16, bits∈{11,13}) 两个宽度,HYLE 整组走自己已 round-trip 验证的 scalar bit-stream 编解码器(`bitstream.h`),其余继续借 Highway。

## 0. 为什么这篇值得独立成文:借库不等于免验

HYLE 的宪章是"借难且已解决的核不手搓"(数学核借 xsimd/Highway、vqsort 借 Highway)。但"借"绝不等于"信"——借来的核**同样要过 HYLE 的对拍闸门**。bit_pack 的对拍恰好是最强的一种:pack 完再 unpack 必须逐值还原,这是 bit layout 自洽的**数学恒等式**,任何一位错位都会被抓。正是这个恒等式在 Highway 一个成熟 contrib 里揪出了满位宽 bug。方法论沉淀:**借核落地时,优先设计"自洽恒等式"型对拍(round-trip / 逆运算 / 已知向量),它不依赖第二实现就能证伪——比"和另一个实现比"更早、更狠。**

## 1. 现象:round-trip 对拍在两个孤立宽度失败

bitpack_test 的核心断言是 round-trip:随机 count 个值(mask 到 bits 位)→ `pack` → `unpack` → 逐值 == 原值。这是 bit-packing 正确性的充要条件。u16 逐位宽扫过去,**只有 bits=11 和 bits=13 失败**,10/12/14 及其余全过。孤立、可复现、与 count/tier 无关——不像随机内存错,更像某个特定 kBits 特化的算法错。

## 2. 诚实记录的误诊排除:先怀疑自己,再怀疑 vendor

借来的成熟库出错的**先验概率低**,所以第一嫌疑永远是自己:
- **疑① 我的 scalar bitstream 编解码器错**?→ 但它对其余 14 个 u16 宽度 + 全部 u8/u32 都 round-trip 净,若它错不会只错 11/13。排除。
- **疑② 我的 block/tail 切分或 packed 偏移错**?→ 同理会跨宽度错,不会只咬 11/13。且 MSVC AVX2 上 block 路径走的是 Highway,tail 走 bitstream,若切分错会在 block 边界普遍露。排除。
- **疑③ MSVC AVX-512/AVX2 codegen bug**(数学核有 `Highway AVX3 在 MSVC HWY_BROKEN` 的先例)?→ 这是最像的嫌疑,必须坐实排除,不能甩锅 codegen 了事(见 §3)。

## 3. 决定性实验:独立探针,零 HYLE 代码,跨 target 复现

写一个**只碰 Highway、不碰任何 HYLE 代码**的 standalone 探针(`scratchpad/hwy_pack_probe.cc`),直接:
```cpp
hn::Pack16<kBits>().Pack(d, raw.data(), packed.data());
hn::Pack16<kBits>().Unpack(d, packed.data(), raw2.data());
// raw 用 FULL kBits-bit mask: (1u<<kBits)-1
```
MSVC `/arch:AVX2` 实测:
```
Pack16<10>: ok    (0/256)
Pack16<11>: FAIL  (14/256)
Pack16<12>: ok    (0/256)
Pack16<13>: FAIL  (102/256)
Pack16<14>: ok    (0/256)
```
**坐实是 Highway 算法本身**,不是 HYLE 也不是 codegen:① 探针无一行 HYLE 代码;② 子代理另在 portable **EMU128** target(纯标量模拟,无任何 SIMD codegen)复现同样失败——EMU128 排除了 MSVC/AVX codegen 的全部嫌疑,只剩算法。与数学核那次"MSVC HWY_BROKEN 是 codegen"是**相反**的结论:那次真是 codegen,这次真是算法,都靠"换 target 复现"区分。

## 4. 根因:上游测试的 mask 笔误让 bug 漏测发货

Highway `bit_pack_test.cc` 造输入(约第 55 行):
```cpp
return ConvertScalarTo<T>(Random32(&rng) & kBits);   // & 11, & 13 ... 字面量位宽数!
```
应是 `& ((1u << kBits) - 1u)`(mask 到 kBits 位)。`& kBits` 把随机数和**位宽数值本身**按位与:kBits=11 (0b1011) 只可能得到 {0,1,2,3,8,9,10,11} 这几个值,远不能覆盖 11 位满范围。于是 Pack16<11>/<13> 特化里那段错位的移位/或**从未被满位宽输入激发**,测试长绿,bug 随 release 发货。这是"测试喂的数据不够宽 → 假绿"的经典陷阱,和 HYLE 自己 simd-08 的 TruncateTo"单平台栈布局侥幸没崩"是同一类:**测试没覆盖到的路径不存在正确性保证,哪怕来自成熟库。**

## 5. 修法:两个坏宽度整组回退 HYLE 自己的 oracle

`bitpack_body.inc`:
```cpp
template <typename T>
constexpr bool bitpack_width_borrowable(unsigned bits) noexcept {
  if constexpr (sizeof(T) == 2) return bits != 11 && bits != 13;  // Highway Pack16<11/13> broken
  else return true;
}
```
对不可借的宽度把 `nblocks` 强制为 0 → 全数组走 `bitstream_pack/unpack`(HYLE 自写、round-trip 验证过的 LSB-first 连续位流编解码器)。pack 与 unpack **共用同一谓词**,所以 layout 恒自洽(不会 pack 用 Highway、unpack 用 bitstream)。其余宽度继续借 Highway(它们已对拍净)。**没有去"修" Highway 的特化**——借核出问题就退回自己的 oracle,不改 vendored 源(改了下次更新 Highway 又冲突);把绕过点和原因就地注释登记。

## 6. 验证

- HYLE bitpack_test:U16 覆盖全 1..16(含绕过的 11/13 走 bitstream)round-trip 净,MSVC BitPack 6/6、simd 全套 213。
- 独立探针(§3 的代码片段即全部逻辑,~40 行,只 include hwy/highway.h + bit_pack-inl.h)可随时重建复跑坐实上游 bug。

## 7. 教训

1. **借核必过对拍闸门**:借不等于信;round-trip / 逆运算这种"自洽恒等式"型对拍不依赖第二实现就能证伪,是借核落地的第一道闸。
2. **换 target 区分 codegen vs 算法**:同一符号在 EMU128(纯标量)上也错 = 算法错;只在某 ISA 错 = codegen。别一上来甩锅 codegen(数学核那次是 codegen,这次不是,靠这招分开)。
3. **测试数据不够宽 = 假绿**:上游 `& kBits` 笔误让满位宽路径漏测——和 HYLE TruncateTo 单平台侥幸同类。喂输入要覆盖被测量的**满值域**,不是几个小样本。
4. **借核出 bug 退回自己的 oracle,不改 vendored 源**:就地注释登记绕过点 + 原因,保留独立复现器。

## commit 序
- bitpack 落地 + 绕过 + 本 hunt 文档:见 simd-09-bitpack 提交。
