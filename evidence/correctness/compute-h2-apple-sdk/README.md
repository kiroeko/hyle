# H2 Apple installed SDK验收

产品最终candidate37744b0ea。初始1989 snapshot与有限input扫描修复分别保留；修复只有私有
目录/热循环，API与其它owner字节未变。原记录不改成新结果，匹配域复用，受影响域重新验证。

- shared/static真实SDK各5/5消费者：Direct权重、混合Graph、合同、原Compute、原Graph。
- 两种SDK各4/4公开头实际独立编译；真正rename安装prefix后配置和运行，原prefix目前仍不存在。
- 无producer源码include回填；真实dylib/static library与mlx.metallib hashes见artifact记录。
- input目录修复后source/shared/static各2/2混合/合同定向通过；Direct/原入口与公共头未变复用。
- genuine Apple-only SDK请求CPU明确拒绝；genuine CPU+Vulkan SDK请求Apple明确拒绝，两项具名diagnostic实际通过。
- Windows无Apple宏/SDK时四头MSVC语法通过。Python detached cmd的参数启动错误发生在编译前，
  改用native PowerShell调用同一cmd后四个实际cl日志通过，不把原启动失败写成PASS。
- Mac重型build1/test1串行；观察RSS最高约648MiB，非OS硬上限。原cache/source对象复用，不clean全库。

[实际批次](sdk-result.json)、[定向修复](input-fix-result.json)、
[当前安装artifact](final-sdk-artifacts.json)、[缺路线](missing-route-result.json)与
[platform-off](platform-off/result.json)给出实际命令和退出码。`.log`仅去行尾空格，原始日志仍在本轮Windows目录。

本证据不证明总物理内存预留、所有模型整机推理、绝对速度赢家或所有硬件都重叠。
