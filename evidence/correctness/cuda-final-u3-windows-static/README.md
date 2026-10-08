# CUDA U3：Windows CUDA-only 静态 SDK 联合资格

独立搬移SDK的普通C++ Direct／Graph已覆盖当前native请求的完整651／649用例并集，
逐名缺项为0；额外普通入口／准入／寿命／reference回归计入后分别为667／652个不同用例，
全部通过、无skip。用例并集见[sdk-test-union.json](sdk-test-union.json)。

## 实际边界

- `compute-cuda`、CPU OFF、STATIC ON、tests OFF的source产品生成并安装，再搬移prefix。
- 消费工程只通过搬移SDK的`find_package(Hyle ... COMPONENTS compute graph simd)`取得
  HYLE头／实现；原ggml reference是独立夹具资产，不从producer获取HYLE实现。
- 实际offline configure／独立link／运行，所需DLL邻exe。完整current／families／artifact／
  普通入口与本地QwenGraph使用仅系统目录的PATH；已纠正reference的fixed组复用同一邻置runtime。
- 源码普通consumer links、两codec纠正与admission实际通过；相同产品的native数学证据复用，
  没有因test-only变化重跑整条651／649路线或P1。
- 保留原559固定GGUF、104身份、34 non-GGUF、完整state／MTP／DSpark／词表和本地Qwen权重。
  [49个family-region关联](family-region-map.json)逐项对应共同builder和native资格用例。

## 复用与补项

完成的Direct N1／N2组复用：产品／vendor库、对应输入与测试正文未改，后续仅改reference寿命。
fixed组纠正后两入口实际各560项；其余未完成current／families按原完整程序续跑。
current按每个完整program独立进程，避免不相关模型尺寸的原库private缓存叠加；没有删stage或改shape。

用例并集核对发现两个Direct admission和五个本地Qwen Graph用例尚未入独立夹具，
通过单独目标补齐，原通过的大executable不重跑。Direct admission无条件包含Linux专属
FlashInfer头的首次编译失败亦留档；复用现有optional-provider helper后，source／SDK各2项通过。

原Q3_K批次的错误reference FAIL没有改成PASS，因果／旧owner回归／纠正资格见
[reference根因](../cuda-u3-reference-codec-lifetime/README.md)。本记录只证明实际完成的边界。

## 尚未完成

Windows shared、CPU+CUDA mixed、Linux shipping矩阵，以及U4 best-of／合法默认、U5清退、U6终审
仍未完成。本记录不签U2／U3／整条CUDA CLOSED。未运行P1，没有性能WIN。

产品及test-only身份、raw日志／XML／哈希见[receipt.json](receipt.json)和`delivery.zip`。
