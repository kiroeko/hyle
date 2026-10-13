# G01 MacBook Air Vulkan 宿主 TSan

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，实际MacBook Air Mac16,12/
macOS27.0.1，未使用Mac mini。[receipt](receipt.json)/[raw.zip](raw.zip)保commands、
logs和三变更TU的真实thread instrumentation。原F5静态TSan树/依赖复用，8路远端编译。

公开F32/F16 D64/128/256驻留数学、effective length/offset、source只读、独立slots、旧
ResultLease/Tensor及释放后重放通过；原生三个反例的精度缓存交错/双顺序、非法D冷
拒绝与零quota回滚也通过。TSAN_OPTIONS=halt_on_error=1，两个实际程序均exit0。

这里只签宿主数据竞争facet，shader/driver/private byte上界不受TSan覆盖，也不签全部
模式任意并发或平台。candidate产品未改，test-only两补例另记输入；未运行项不改PASS。
三交付/宿主san结果仍不等于整G01完成，CPU交付/风险、P1和最终review继续待验。
