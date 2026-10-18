# G03 GP01/GP02 四份 relocated Compute SDK

固定candidate `b42832de3ddda21bdc3e98ad4cc87e73622cbba4`，实际Air；Apple/Vulkan各
static/shared，[receipt](receipt.json)保真实安装/搬移/外部configure/build/run、身份与
compdb，[raw](raw.zip)保完整日志/runner/输入。四份SDK全部PASS，累计58.886s。

两个consumer完全复制canonical原字节，无oracle替换、private header、产品源码include
或故障hook；只find_package(Hyle COMPONENTS compute)、只链hyle::compute。每份
public CPU64与本route native64（Apple另BF16共128）实际运行；现有unique/readers/
native uses、retained result/submission与delta bytes证明沿同一公开源码。
四个独立头是batch_transfer、paged_arena、对应memory与batch；四SDK共16 standalone
TU通过。compdb核实仅relocated include，不从producer source取得头。

[Windows当前native public compile/link](windows-compile.log)同cached cpu;vulkan树，
MSVC、四jobs、78.109s；[记录](windows-compile.json)保确切命令。只编译链接，未执行
exe/GPU，不代签Windows数学或原G02重型未决门。候选普通源码CPU/native/failure与
六host san分别有独立证据，不由这个SDK记录代签全CUDA producer链/整G03/goal。
