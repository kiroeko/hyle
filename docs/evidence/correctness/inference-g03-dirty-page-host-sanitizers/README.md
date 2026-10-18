# G03 GP01/GP02 六组宿主 sanitizer

固定candidate `b42832de3ddda21bdc3e98ad4cc87e73622cbba4`；实际Air身份、每组source
SHA/command与全部选定TU的真实instrumentation flags见[receipt](receipt.json)，全部
原日志/runner/输入9及验证头在[raw](raw.zip)。Apple/Vulkan分别ASan、UBSan、TSan，
cached trees四jobs串行；每组执行同一公开native192域的本route64或128步、actual copy
九风险的本route4/5以及public CPU64。六组final全部PASS。

ASan显式detect_leaks=0；仅签host，不签LSan、shader/driver或private pools。TSan借原
[G02独立反证与显式模型](../inference-g02-native-page-joint-host-sanitizers/README.md)：
LLVM21原fence未建模，validation-only final-counter acquire头（原release RMW与中间
drop不变）；Apple另借原同Metal command-buffer commit/completion模型。compdb核实
实际模型include与thread flags；产品/cache/util未修改，无suppression。不重做已封存的
纯C++/Metal独立反证，也不冒充无模型TSan结论。

首configure因上传清单漏了新CPU consumer而失败4.016s，补精确input9后同tree续跑；
原FAIL、elapsed/cumulative保留。各route64数学、F2 actual成功/unknown/error与guard
退休，以及old result阻止更新的资格只对应固定有限域。SDK交付和完整pending producer
链尚未签，不签整G03/goal。
