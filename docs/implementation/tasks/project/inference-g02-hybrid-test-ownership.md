# G02 影响验证的 hybrid 测试归属修正

2026-10-13，test-only维护。baseline337422e05，G02产品candidate26d7c2cb7不变。
selector实际选择CPU＋CUDA/Apple/Vulkan直接消费者，但对应tests owner片段缺失，
tests/CMakeLists.txt按既有合同拒绝配置。既有sample、contracts与A有限shape/control
消费者已存在；登记到真实owner的aggregate/CTest label，不复制新数学或重写构建/checker。
支持的OS/route执行真实消费者，不支持时只保aggregate并保selector skip reason。
仅改三个tests owner CMake片段，产品owner不重开；验证为恢复同树配置及对应actual gates。
所有原配置FAIL、SDK路径修正与累计耗时保留，不能将恢复配置签成产品数学PASS。
