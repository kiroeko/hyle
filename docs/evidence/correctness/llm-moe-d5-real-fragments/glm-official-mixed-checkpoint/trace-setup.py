from pathlib import Path
import shutil

root = Path.cwd()
out = root / '.scratch/validation/d5-f40-trace'
out.mkdir(parents=True, exist_ok=True)
source = root / 'tests/compute_route_cuda/d5_qwen38_official_test.cc'
original = source.read_bytes()
assert b'D5GlmOfficialMlaDiagnostic' not in original
(out / 'original-test.cc').write_bytes(original)
shutil.copyfile(root / '.scratch/b-cuda-d5/Testing/Temporary/LastTest.log',
                root / '.scratch/validation/d5-f40-glm-mixed/cuda/native-tests.log')
insert = r'''
TEST(CudaMoeFragments, D5GlmOfficialMlaDiagnostic) {
  namespace glm = d5_glm_remaining_fragments;
  auto fixture = glm::mla(glm::kCases[3]);
  const auto dump = [](const std::string &name, std::span<const float> values) {
    std::ofstream stream("C:/public/project-hyle/hyle/.scratch/validation/d5-f40-trace/" + name,
                         std::ios::binary);
    stream.write(reinterpret_cast<const char *>(values.data()), values.size_bytes());
  };
  for (std::size_t r = 0; r < fixture.replays.size(); ++r)
    for (std::size_t p = 0; p < fixture.outputs.size(); ++p)
      dump("r" + std::to_string(r) + "p" + std::to_string(p) + "-expected.f32", fixture.replays[r].outputs[p]);
  fixture.observe = [dump, r = std::size_t{0}](std::size_t p, std::span<const float> values) mutable {
    dump("r" + std::to_string(r) + "p" + std::to_string(p) + "-actual.f32", values);
    if (p == 5) ++r;
  };
  run_q24_semantic_fragment(std::move(fixture));
}
'''
marker=b'TEST(CudaMoeFragments, D5GlmOfficialMixedRemaining) {'
assert original.count(marker)==1
changed=original.replace(b'#include <algorithm>',b'#include <fstream>\n#include <algorithm>',1)
changed=changed.replace(marker,insert.encode()+b'\n'+marker,1)
source.write_bytes(changed)
script = (root / '.scratch/d5-kimi-owner-windows.py').read_text(encoding='utf-8')
script = script.replace('d5-kimi-owner', 'd5-f40-trace')
script = script.replace("['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda', 'compute_route_cuda', 'graph_route_cuda']", "['compute_route_cuda']")
script = "import os\nos.environ['GTEST_FILTER']='*MoeFragments.D5GlmOfficialMlaDiagnostic'\n" + script
(root / '.scratch/d5-f40-trace-run.py').write_text(script, encoding='utf-8')
print('Diagnostic prepared; original failure log preserved.')
