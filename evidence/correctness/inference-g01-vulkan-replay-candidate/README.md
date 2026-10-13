# G01 Vulkan RR 候选：交付与宿主风险

不可变candidate `a40003f7edb54a3da3113d0cd401b46211e0b0f0`。[receipt](receipt.json) 与 [raw.zip](raw.zip) 保存实际
命令、elapsed、日志、原插桩compdb和canonical单向输入身份。产品文件SHA逐项对Git
candidate核对；实现阶段原FAIL仍在前一份[stage证据](../inference-g01-vulkan-replay-stage/README.md)。

实际MacBook Air Mac16,12上source公开/native测试、真实改名搬迁的static/shared SDK
普通消费者均通过。SDK仅find_package(Hyle)+公开Graph/Compute，独立consumer不引用
HYLE private API；测试oracle bit carrier的复制不变更数学。覆盖72个原数值片段、
64步offset/backing/owner重绑、独立slots/旧输出寿命，以及65-element行间padding。

同Air宿主ASan、UBSan、TSan三个构建分别通过上述公开三项及四个native风险tests，
包括128次原生更新/owner切换、strict/normal交错、非法D与零quota回滚。每种compdb
核对ggml-vulkan.cpp、native_device.cc、region.cc和两个测试TU的实际插桩；private
runtime/work seam由这些TU编译。ASan禁用平台不可用LSan，不签shader/driver。

Windows RR相关目标4路编译通过，未运行本机GPU；原candidate Windows ASan收据
仍保持原身份，未假装签发RR版本。正式RR同合同P1另跑，整G01和总goal仍未完成。
