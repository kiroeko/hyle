# Y28：Vulkan 受检 I32 Scan 的四构建回归

候选`f0115f967f0e4ad302778ecd909b7f918d7c7845`，baseline `212765cf`。
本记录仅签I32 Scan及selector选中的Direct/Graph正确性回归，包含已有F32 Scan；
不是完整C1、Scan三交付或P1。Provider产品未变，其独立factor按selector跳过。

| 环境 | Direct / Graph | 结果 |
|---|---|---|
| Windows strict static | 75 / 23 | 98 PASS |
| Windows strict shared | 75 / 22 | 97 PASS |
| MacBook Air Mac16,12 static | 75 / 23 | 98 PASS |
| Mac mini Mac16,10 static | 75 / 23 | 98 PASS |

共391项。首行Windows先行8个整数Scan和1个Graph定向用例通过，不再重复计入总数。
Windows两构建的source/MASM检查各2项通过。没有产品纠正，也没有P1。
Windows/Air启用validation，日志没有VUID或同步错误；Mini未安装该层，仅签正确性/可移植性。
所有构建parallel2、重型验证串行，Windows DLL邻置、PATH unset、SetErrorMode(0x8003)。

## 实现和独立验证

- I32以两个U32字完成精确I64局部树形前缀、递归block总和与carry传播；现有shader构建检查只
  允许Shader capability，不依赖shaderInt64、subgroup或全局自旋。最终全局前缀检查与条件Copy
  分开，失败不覆盖整个算子输出；与Gather复用status/原生资源/完成通道，用bit2报告前缀越界。
- 独立I64逐项参考验证inclusive/exclusive、宽0/1/127/128/129/255/256/257/8193/65535/65536/
  65537、多行、任意/负轴、rank8、奇数/零stride和0–4字节起点。临时block和超I32但全局合法
  的正负构造成功；正负溢出、越界后抵消及exclusive最后总和越界都返回OutOfRange。
- 固定Recorder验证最后一行溢出时所有输出保持哨兵值，含恢复非末轴的布局；输入保护和alias
  拒绝保留。失败槽退役、新槽恢复、quota耗尽回滚、五次重放无payload分配、close后旧结果可读。
- Graph的counts→exclusive Scan→原版ggml Gather→RmsNorm有三个发布输出；两槽/五次变化输入、
  独立整数/位级Gather/FP64 norm参考、坏前缀完成失败及新槽恢复均通过。原Graph和Program句柄
  在运行前销毁；捕获表和旧结果保持有效。既有OOM/部分接纳/drain与F32 Scan同候选回归通过。

整数实现是已审计的缺口补齐，不能计作新增成熟Provider；F32继续原版ggml cumsum。
最多四层前缀、九次计算dispatch，额外布局Copy分别计费；主要scratch为8N宽前缀、4N结果，
以及递减的block总和/前缀和status。未比较性能，不声称默认胜出。

[聚合结果](result.json)、Windows原始包和两个Mac receipt/原始包绑定源码、命令、二进制摘要及
实际测试数；Windows包保留selector的115个skipped/reused原因。统一脚本
[47,955行/194个实现文件](implementation-loc.json)，含GLSL、不含测试/bench/第三方，自检通过。

归档核验后清理[Mini上一份验证源码](y28-mini-source-cleanup.json)87,197,356字节，以及
[Windows旧输入archive/bundle](windows-old-input-cleanup.json)36,622,474字节，共123,819,830
逻辑字节。当前源码、构建/依赖cache、SDK和验收证据保留。

尚未签收：Scatter、Scan source/static/shared实际消费、完整C1真实片段、优选淘汰和global
package门。CPU oneDNN资格BLOCKED不变，共同矩阵不因这391例增加已验收行。
