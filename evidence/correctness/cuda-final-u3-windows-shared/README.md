# CUDA U3：Windows CUDA-only shared 交付资格

源码shared产品与搬移后的独立SDK已完成configure／build／install／offline link／实际加载，
DLL邻exe，SDK运行PATH仅为系统目录。新交付检查Direct **166项**、Graph **151项**通过，无skip。

本批验证shared ABI／visibility／资源与加载边界：每种实际GGUF角色／编码配对60项、
全部34个non-GGUF packed合同、全部必要模型族program／当前官方与三sidecar完整program、
本地Qwen真实权重及普通入口／负空间／两slot重放。保存的用例见[sdk-test-union.json](sdk-test-union.json)。
并未把60项称为全559形状矩阵；相同产品数学的两平台完整651／649资格与
[static SDK完整并集](../cuda-final-u3-windows-static/README.md)按来源／算法未变复用。
Source普通入口／纠正reference／本地QwenGraph另有实际smoke，不从link成功推断运行通过。

首次source shared邻置COPY_FILE报output Permission denied：同目录并行writer冲突。
独立[tool task](../../../docs/implementation/tasks/build/runtime-staging-directory-writer.md)
复用原Apple同形目录锁后，实际j2构建从保留对象续跑通过；source与installed helper的
初次／DLL-only更新／无变化、双consumer／alias／conditional／empty合同亦通过。
原失败日志没有覆盖或改成PASS。

精确product／test／tool身份、原日志／XML与SHA见[receipt.json](receipt.json)和`delivery.zip`。
没有改变原数学／误差门，没有P1／性能WIN。CPU+CUDA mixed、Linux shipping、U4／U5／U6仍待。
