# Native ragged 比较器预检

[receipt](receipt.json)与[raw](raw.zip)保存两个比较器三形状编译/oracle/readonly/tail预检、
所有实际失败与成本诊断。只有diagnostic，不是正式P1/非劣或CLOSED声明。

Apple peer直接调用固定原MLX fast SDPA，通过原Context.compile/Program/两slots/ready
packed sources/checkedCompletion/owning packed result，与candidate同完成边界。Vulkan
peer独立原full-F32 flash shader recipes按request组合到原Program，保mask/转换/调优/
split-K，输出拼接借原storage算子。共享F16数据与独立double oracle原bound2e-3，未放宽。

Apple预检首次compile暴露opaque default构造与MLX array不支持default构造，改为原
工厂initializer/optional；随后borrow者晚于借用池销毁导致实际SIGSEGV，显式先释放
slots/arrays/program后通过，原product candidate280不改。这个失败没有被写成PASS。
公开MemoryResource default ctor的incomplete-type反例归现有G10/交付接口待验，不
另加全局调研轮，当前候选性能完成后再按真实reopen trigger有限修正。

实际诊断process成本Apple2.361s、Vulkan2.454s；Vulkan短留出约13%差距需原A/A、
双execution/allocation、placement与独立进程采样确认，不能由单次数据改公共API。
正式前固定16paired rounds/min1/4ms、n15/31及5% margin/原controls；正式结果另存。
