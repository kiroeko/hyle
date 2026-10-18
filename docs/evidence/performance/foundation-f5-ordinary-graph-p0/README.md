# F5普通Graph默认字段与warm codegen P0

F4固定29d3→F5固定25e。actual WSL GCC14.2 shipping/static CPU+CUDA的原迁址SDK archives
只读提取，未新建延迟benchmark或依据地址/纳秒改API。普通InvocationState与
LeasedInvocationState完整声明逐字相同，公开对象只加friend，无字段。新batch/coordinator/
mirror/page footprint状态只在显式opt-in对象；普通Graph不调用新增cold访问visitor。

原executable.cc.o与候选整ELF对象sha256完全相同。六个普通submit符号size均相同：
三个Invocation 456/511/246字节，三个LeasedInvocation 553/1830/435字节。
五个方法raw机器码hash相同；span版Leased submit唯一六个不同字节来自三个相对call
到同名record_error helper的位移，去掉这三处imm32后其他全部字节相同，helper raw hash
本身相同。不是把不同逻辑代码归一化掉；保存原bytes、完整反汇编、call offsets与所有hash。

这支持本匹配配置下无新增普通字段/分配/plan调用/warm指令税；不声称所有编译器每字节
相同、吞吐WIN、capture或native driver成本0。F5整体算法/cold metadata/reference显式
搬运/private UNKNOWN费用及自有LOC仍另核，不能用这个局部门代签全部P0。

证据、原immutable artifact路径/hash和源字段proof见[receipt](receipt.json)/raw.zip。
工具只私有定向读ELF，没有重写公共性能统计核心或运行P1。
