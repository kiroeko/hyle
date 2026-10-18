# G01 Vulkan：驻留端口实现阶段

[receipt](receipt.json) 与 [raw.zip](raw.zip) 保留原3个Copy反证的FAIL、测试API拼写
编译FAIL以及最终实际通过。没有修改shader、数学或统一仪器。

Air真实source的9项公开测试通过：原四项managed/mixed-provider回归，72个原attention
数学片段、readonly/独立slot/旧输出、64步offset/backing/owner重绑，合法容量view
native_dispatch_count从3变0。65-element非vec4 padding仍明确2个KV Copy且数学通过；
相同descriptor但不同byte origin的坏binding提前拒绝，随后的对齐binding执行成功。
Provider在cold graph记录实际native port对齐，原资源失效/真实退休机制不变。

Windows仅两个相关TU及消费者增量编译，最多4路，未运行本机GPU。最终源文件SHA
保存在收据，canonical输入单向验证；三交付、宿主san、正式P1尚未运行，不记PASS。
原a400合格P1 FAIL另存，不因复制次数变0就推导性能通过或签整个G01完成。
