# G03 GP01/GP02 Windows public CPU 源码阶段

2026-10-14，AUDITED draft；四owner/八产品文件实际SHA在[receipt](receipt.json)，
MSVC19.50.35728.0、原缓存tree、四jobs。只build两个定向targets和轻量CPU执行；
没有GPU/full model run。公共consumer只链hyle::compute。

[64步](host-cpu-public-probe-2.log)每步1024 dirty/16384 page字节、原generation/offset、
所有untouched rows与独立数值reduction通过。耗尽arena metadata quota后仍能取得原ready
writer，未重新allocate/复制；旧shared frame、admitted use与retained submission阻止
取回写权限，成功submission也须drop后才可unbind。更新后的原page再次组成精确frame。

[九个arena/F2/F4测试](paged-arena-risks-2.log)全部通过，包括七个原直接消费者与两个
新增回归。foreign/subrange/closed、失败不消费、cold撤回、重复unbind、prepared失效、
output已publish拒绝和原F4 sharing/actual guards保持。新copy withdrawal只借原私有
CopyStreamAccess；没有更改success/retirement/coverage或数学。

[raw](raw.zip)保全部attempt/log/最终源码/runner及两次fixture失败输入：首次错误地用
Result宏解包optional quota reservation，第二次把整页view读进1byte目标；分别修fixture
的返回类型处理与明确subview，产品35行未改变。receipt保累计时间，不把原FAIL覆盖。

这是源码CPU阶段，尚未签relocated static/shared、host san、actual Apple/Vulkan dirty
页→数学与failure/unknown，也不签完整pending producer→attention、G03整项或goal。
