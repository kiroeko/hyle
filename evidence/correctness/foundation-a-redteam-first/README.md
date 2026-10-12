# Foundation A 候选前反证

源码反证发现 Apple 未限制 mandatory-alias custom；共同 NativeAnalysis 增加活 GPU 的
资格拒绝，CPU 仍由原 artifact alias_support 裁决。新增反例及原共同测试41/41 PASS。
另外修正 pure control MLX diagnostics，不把只有 MLX 的图报告为实际含 ggml model region。

[公开 consumer](../../../tests/graph/public_effect_consumer.cc) cancellation：原 consumer CPU
callback 写入以后，以测试进度 gate 阻塞，caller 在有限2秒内观察开始、请求取消并释放 gate。
旧写入41仍存在、GPU/read后继不开始、返回 Cancelled；Win Vulkan source PASS。
该 gate 仅为 consumer progress 仪器，不是新产品线程/reaper/效果或 rollback；真实业务 state
和 observation 的读写均有独立资源 ports/footprints。异常释放也允许原工作退休。

原失败预算继续沿 [第二批](../foundation-a-effects-first/README.md) 累计。这里没有签 Apple
最终 snapshot、sanitizer、installed SDK 或全部工程门；这些仍需候选定向资格。
