# G01 普通 CPU 驻留 consumer 三交付

不可变 candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`。同一公开 consumer 在 canonical 源码 target、实际
relocated static SDK 与 relocated shared SDK 均退出 0。[receipt](receipt.json) 与
[raw.zip](raw.zip) 保留所有实际命令/耗时/失败、输入和安装产物身份。

普通 `cpu::create_context` 在 BF16 GQA8/2 D256 上实际选择成熟 ggml；两种 layout、
Q=1/17、有效 KV31/容量37、causal bottom-right 数学、padding 哨兵、两个输入 seed、
独立输入 buffer、slot 重放及 K/V 源只读均通过。BF16 用公开 uint16 storage 与独立期望值，
仅 include 安装的 HYLE public headers，没有 expert factory、SIMD 或私有 product source。
这只证明有限 resident consumer；不代签全模型、任意 page table 或整体 G01。

旧树遗留验证 hook 与旧 build-local ggml symlink 导致的 configure FAIL 均保留；
隔离旧 dependency copy，按当前认证 source cache 重新生成，没有修改共享 source cache、
candidate 产品或公共构建工具。shared/static 使用同一 producer tree，先完成共享安装再
切静态，运行消费者之前实际改名 staging 为 relocated。Windows 编译最多4 jobs，无 GPU run。

CPU 三种宿主 sanitizer 另见对应记录；同合同 P1、LOC/终审及多请求消费仍待，goal 保持执行。
