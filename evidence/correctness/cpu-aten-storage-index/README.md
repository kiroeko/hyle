# CPU ATen 存储/索引资格

当前为原 API diagnostic 与有限审计，产品正确性和交付尚未运行。

- baseline `79ded44e1307b5161f3068b65b22b9feb894d029`，scope `7d67b3fc858f25809d6a9cf227429f3c700335ae`。
- 已锁 LibTorch 2.8.0 CPU Windows SDK；original-manifest.json 为源码、命令、日志和 native DLL 身份。
- 原 Byte `where_out`/`index_select_out`/`copy_` 实测 1/2/4/8 字节载体、odd-byte stride、
  condition 非零、I32/I64 index、重复读取与 rank9；empty 与负 index 原 API 反例保留。
- 原 stable sort + neighbor mask + masked_select + index_select + index_copy_ 程序实测
  1/2/4/8 字节、重复索引，最终按最后逻辑更新取值；没有执行原 duplicate index_copy 的未定义并发写。
- 共454个原程序断言通过；这不是454个算子、模型或普通入口 PASS。
- 原 SiLU/GELU 的 -Inf 给 NaN、Sigmoid -100 给0、amin在具名混合signed-zero输入给+0。
  与现存公共数值边界有差异，因此此批不准入这些 math，不进行 Native 性能比较。
- 固定 SI1–4：独立 factory / Byte 原存储程序 / checked生命周期 / 普通默认接缝与批量资格。
  ScalarValue 只在冷阶段建立一个 scalar 的 exact bytes；不新增 payload 数学或自己的 copy kernel。
- Assign 对任意 public index rank 先原 reshape/flatten，再稳定选取唯一胜者；native packing与排序成本
  保留。checked index预扫描是边界检查，所有输出写入之前完成。

三交付、Linux、普通 Direct/Graph、资源/alias/OOM gate 均 UNPROVEN，待同一稳定产品候选运行。
