import ctypes
from pathlib import Path
import subprocess

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
root = Path.cwd().resolve()
bin_path = root / '.scratch/build/m2-vulkan-indexed/bin'
out = root / '.scratch/validation/d5-f27/correction/extra'
out.mkdir(parents=True, exist_ok=True)
for name, filter in (
    ('hyle_graph_route_vulkan_test.exe', 'VulkanGraph.OriginalFloat8RegionContinuesIntoGgmlAndNativeCopy:VulkanGraph.CheckedScanChainsTwoOriginalProgramsWithIndependentResults'),
    ('hyle_compute_provider_iree_vulkan_test.exe', 'IreeNative.*'),
    ('hyle_compute_route_vulkan_test.exe', 'VulkanLifetime.PartialAllocationFailureReleasesNativeQuotaAndDescriptors')):
    log = out / (name + '.log')
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(bin_path / name), '--gtest_filter=' + filter],
                                cwd=bin_path, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
    print(name, result.returncode)
    print('\n'.join(line for line in lines if '[  PASSED  ]' in line or '[  FAILED  ]' in line or '[  SKIPPED ]' in line))
    if result.returncode:
        print('\n'.join(lines[-40:]))
        raise SystemExit(result.returncode)
