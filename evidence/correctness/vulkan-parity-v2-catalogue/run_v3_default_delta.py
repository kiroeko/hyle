"""Serial original IQ1 source qualification; hidden children, adjacent DLLs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path('C:/hv/vp1/bin');OUT = Path(__file__).resolve().parent
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
FLAGS = {'creationflags': subprocess.CREATE_NO_WINDOW}
ENV = dict(os.environ, HYLE_VULKAN_VALIDATION='1', HYLE_TEST_VULKAN_VALIDATE='1',
           VK_LAYER_PATH='C:/public/tools/hyle-vulkan/1.4.357.0/Bin')
ENV.pop('HYLE_VULKAN_IQ1_DELTA', None)
candidate = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,**FLAGS).strip()
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True,**FLAGS).strip():
    raise RuntimeError('clean immutable candidate required')
tag = 'v3-default-' + candidate[:8]
ENV['HYLE_D5_QWEN35_FIXTURE'] = str(ROOT / '.scratch/d5-qwen35-six-profile-fixture')
jobs = [
('direct-default','hyle_compute_route_vulkan_test.exe',
 'VulkanManaged.ColdRulesMatchFullConditionAxesAndUnknownDefault:VulkanManaged.DirectNormRebindsIndependentSlotsAndRetainsOldResults:VulkanManaged.PointwiseFamilyUsesRealProviderWithoutNormAnchor:VulkanManaged.ElementwiseDeviceLimitsRetainQualifiedFallbacks:VulkanManaged.PackedAndSeparateSwiGluRetainHighRankRowHalves:VulkanManaged.D5RealQwenProjectionUsesExactProfileAndReplays:VulkanIree.OriginalIq1ExpertProgramsAndChangedInputs:VulkanIree.OriginalIq1IndependentLeaseCancelCloseAndOom:VulkanIree.OriginalOfficialGroupedFp4Fp8SharedDistinctAndChecked:VulkanIree.OriginalFloat8DirectComposesBFloat16WithoutExpandingInputs'),
('graph-default','hyle_graph_route_vulkan_test.exe',
 'VulkanGraph.ManagedLayerNormKeepsMisalignedCapturedRowsAndPublishedForks:VulkanGraph.ManagedNormFusionPreservesPublishedForksAndNativeBoundaries:VulkanGraph.PointwiseRegionPreservesScalarOrderPublishedValuesAndRebinds:VulkanGraph.StableModelGatesShareMatureRegionAndKeepPublishedForks:VulkanGraph.B2BatchedLinearAndHadamardShareOneMatureRegion:VulkanGraph.D5RealQwenProjectionUsesExactProfileAndReplays:VulkanGraph.OriginalIq1ExpertProgramsAndChangedInputs:VulkanGraph.OriginalIq1IndependentLeaseCancelCloseAndOom:VulkanGraph.OriginalOfficialGroupedFp4Fp8SharedDistinctAndChecked:VulkanGraph.OriginalFloat8RegionContinuesIntoGgmlAndNativeCopy')]
receipt = {'candidate': candidate, 'steps': [], 'scope': 'V3 cold conditions and legal unknown default, actual managed/IREE Direct/Graph chains and replay/resource delta only'}
failed = False
for label, name, selected in jobs:
    exe = BUILD / name;dlls = sorted(BUILD.glob('hyle_*.dll'))
    if not exe.exists() or not dlls: raise RuntimeError('EXE or adjacent DLLs missing')
    log = OUT / (tag + '-' + label + '.log');xml = log.with_suffix('.xml')
    if log.exists() or xml.exists(): raise RuntimeError('candidate already attempted, no overwrite')
    print('START ' + label, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([str(exe),'--gtest_filter=' + selected,'--gtest_output=xml:' + str(xml)],
            cwd=BUILD,env=ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=1200,**FLAGS).returncode
    raw = log.read_text(errors='replace');validation = 'Validation Error' in raw or 'ERROR / SPEC' in raw
    stats = ET.parse(xml).getroot().attrib if xml.exists() else {}
    print('END ' + label + ' exit=' + str(code) + ' tests=' + stats.get('tests','?'), flush=True)
    receipt['steps'].append({'name':label,'returncode':code,'filter':selected,'tests':stats,
        'validation_error':validation,'log':log.name,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'xml':xml.name,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
        'adjacent_dlls':[p.name for p in dlls]})
    if code or validation or not xml.exists() or stats.get('disabled','0') != '0' or stats.get('tests','0') != '10' or '[  SKIPPED ]' in raw:
        print('\n'.join(raw.splitlines()[-25:]),flush=True);failed=True
    (OUT / (tag + '-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
raise SystemExit(1 if failed else 0)
