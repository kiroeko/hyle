# G02 实际联合64步复用

原fixed product1cd不变，test-only将普通joint consumer的每组12episodes改为64。
actual Mac16,12/M4/同原source cached cpu+Apple/Vulkan trees，四jobs串行。Apple F16/BF16
三D×causal/noncausal十二组768episodes；Vulkan F16六组384，共1152 final episodes。
每组仍是实际CPU ABS/Cast→原page attention→CPU Cast/Neg，独立double全输出、单请求/
混合两请求、固定active输出的layout演化、两slots/旧result、精确frame与32byte真实table
费用全部PASS。原12步证据/失败保留，不改oracle/形状/产品来取得PASS。

[receipt](receipt.json)、[Apple](apple-final.log)、[Vulkan](vulkan-final.log)、[raw](raw.zip)
保实际28输入SHA/命令/原日志；48.188s。只扩复用时长，SDK/san/Windows原已验接口与
产品bytes不变，不为循环常量重复所有门。产品候选不重置，performance不重复。

[LOC](loc-owner-union.json)是原loc-report/cloc的fixed candidate1cd self-check，边界明确为
四owner的src并集及共用controller头；不是整个Graph目录或单独graph_batch的LOC。
