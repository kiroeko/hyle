# G01 actual Vulkan 宿主地址 ASan

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，复用Windows/MSVC静态ASan树，
原partial prefix/elapsed/build FAIL保留。[receipt](receipt.json)、[raw.zip](raw.zip)保存
实际commands、input hash及Ninja compdb；变更region和两个consumer TU均有地址插桩。

public resident consumer全部数值/有效区/offset/source只读/旧结果和重放通过。native风险
消费者另过三例：strict/normal scalar缓存双顺序隔离、未准入D8/72/512执行前拒绝、零
quota实际graph buffer分配失败返回ResourceExhausted且quota归零。后一项不声称全部native
host allocation点已注入，也不代签opaque private总字节上界。

首次链接发现GTest未插桩而产生annotate_string/vector不匹配，沿同树给GTest启用原ASan
配置后通过；没有关闭annotations、弱化异常退出或改变candidate产品。consumer变更仅
追加两个test-only风险case，实际test输入hash另记录，不冒充旧candidate测试字节。

ASAN_OPTIONS为halt_on_error=1:detect_leaks=0；MSVC无LSan，shader/driver不受宿主ASan
覆盖。这里只签地址facet和具名caller/native边界，不签所有san、性能或整个G01结束。
