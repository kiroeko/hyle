# Native page 宿主 sanitizer 阶段

产品 candidate `520027a57`，actual MacBook Air Mac16,12/M4/arm64、macOS27.0.1
26A434、AppleClang21。Apple/Vulkan 的 ASan、UBSan、TSan 六组宿主验证均取得最终
退出码0；每组实跑公开分页数学及六类 actual native 风险/空域/冷分配失败。只构建
对应 route 与两个直接消费者，四 jobs 串行，全部构建与运行累计480.558秒。

Apple TSan 结论有明确条件。两次原始运行在原 metal-cpp 的 completed-handler
交接处报 race；第二次已包含 MLX 宿主 instrumentation。独立、无 HYLE/MLX 的
Metal blit＋512次回调程序复现同一报告。最终验证只在可丢弃的 metal-cpp 头副本中，
在真实 command buffer 的 commit 前标记 release，在同一个完成回调访问 handler 前
标记 acquire，表达 Metal 原生提交到完成的同步关系。该独立程序与实际消费者随后
通过。没有 race suppression、公共产品改动或原 dependency cache 改动，不能写成
未经标注的原始 TSan PASS。原两次失败及独立探针全部尝试保留在 raw.zip。

receipt.json 保存六组命令、实际硬件/编译器身份、全部退出码/elapsed、原始失败及
边界前后 SHA。raw.zip 保存逐组日志、原 test parent、已核 sanitizer flags 的实际
编译命令、最小探针与验证头副本。ASan 使用 detect_leaks=0；这些结果只签宿主对象、
owned metadata、共同 controller、ready-page guards 和原生失败退休，不签 LSan、
shader、driver 或 private native pool instrumentation。独立数学输出校验另有 source
与 SDK 阶段证据。

五 owner 仍 CANDIDATE；同合同不可变 P1、联合分页及整个 G01–G10 尚未完成。
