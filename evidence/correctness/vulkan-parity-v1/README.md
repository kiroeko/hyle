# Vulkan V1：原格式接线与真实数值程序

baseline a19391c7e。共同目录559合同的
[原source对账](source-binding-audit.json)只记录静态事实：119合同8格式有原Vulkan body但
未绑定、8合同3个扩展IQ1没有当前source资格；五种实际embedding类型对应8个漏绑定合同。
全部runtime初始UNVERIFIED，不从CPU/Apple PASS推导GPU能力。

产品`3b7d3ba9`仅补ggml Vulkan原type/embedding映射，原资源检查/实现不改。原producer
一次configure/build2通过，真实primary RTX4070/617.14/SDK1.4.357；DLL邻置。
首批`b2a66c57`[Provider25/25](native-b2a66c57-provider.xml) PASS，
[Direct3/5](native-b2a66c57-direct.xml)、[Graph2/4](native-b2a66c57-graph.xml)。
八embedding与其它格式通过；四个测试失败全在IQ1_S数值。

锁定backend的IQ1_S/MMVQ先将F32 activation转为Q8_1，d/s存F16；旧helper按无该转换的
F32方程。一次[c9419820捕获](iq1-cause-c9419820.log)：原独立Q8_1与IQ1_S整数dot/
rounded-s公式最大误差4.47e-7/8.94e-7，旧F32参考差0.004785/0.009571。捕获不是正式
模型PASS。原失败保留，不归因为provider漏接或盲改kernel/容差。

测试修正对实际原程序使用独立原Q8_1方程和更紧1e-5+2e-6relative门，另保留既有共同
GGUF归一化模型量化门对未量化activation数学期望检查。预先由device/source事实绑定，
不依据实际结果择近；输入/权重/output不改，其他17格式原门不变。其编译首次漏Graph
test私有include路径；03cfe0f3只纠正相对include，未扩公开产品依赖或重配置。
正式IQ1_S四失败ID两入口delta待验，明确打印type；通过的provider/embedding/其它格式
不重跑。119/8合同全部actual geometry与全新模型分支仍留父轮矩阵，微型资格不代签。

[串行驱动](run_windows_batch.py)只运行该批实际GPU程序，CREATE_NO_WINDOW和ErrorMode
禁止缺DLL/崩溃弹窗；EXE旁保有CMake-stage DLL，validation layer显式进程级启用。
原route/Graph性能BLOCKED未解除，本批不P1、不self扩张、不重写公共验证系统。
