# G03 精确 dirty 页更新的公共形状反证

baseline `3cbbd4e2e`，2026-10-14，Windows MSVC、既有 CPU backend，公开
`hyle::compute` consumer；只运行短 host copy，不运行 GPU/模型。源码与构建/run、实际
耗时在 [raw](raw.zip) 和 [receipt](result.json)，[公开调用](public-probe.cc)可复现。

整页16384字节 copy→F2 successful output→PagedPageWriter.complete通过；同一page16、
GQA8/2、D256中，仅dirty一行1024字节后，F2全页output合法拒绝“uncopied logical bytes”，
partial output又被page writer合法拒绝“not the original page writer”。原数据已明确初始化，
仍没有公共mutable return供caller保留未写部分并完成原page。此证据说明组合形状缺口，
不把两个正确拒绝当作F1/F2权限bug，也不授权扩大F2 output覆盖。

[固定peer身份](peer-inputs.json)及三个实际producer调用片段分别是
[SGLang](sglang-producer.txt)、[vLLM](vllm-producer.txt)、[llama.cpp](llama.cpp-producer.txt)。
它们只更新loc/slot_mapping/row IDs指定的数据，未要求每步复制整个KV历史。参考的模型/
cache政策不下沉；这些调用本身不证明HYLE pending producer→native attention已经完整。

现有内部CopyStreamAccess::take_mutable已为F3证明相同的写权限回收；本轮可以复用原
retirement和成功门，形成普通增量consumer的公开入口。PagedArena还需从其原unique ready
page进入writer，以保既有内容/同generation，而不是释放后重新allocate一个无内容合同的
block。后续测试必须检查reader/actual uses拒绝、失败不消费、dirty bytes和原页范围。
