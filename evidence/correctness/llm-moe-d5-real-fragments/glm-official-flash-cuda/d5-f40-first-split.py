from pathlib import Path

root=Path.cwd()
out=root/'.scratch/validation/d5-f40-first-split'
out.mkdir(parents=True,exist_ok=True)
path=root/'tests/compute_route_cuda/d5_qwen38_official_test.cc'
original=path.read_bytes()
assert b'D5GlmOfficialMlaDiagnostic' not in original
(out/'original-test.cc').write_bytes(original)
changed=original.replace(b'#include <algorithm>',b'#include <fstream>\n#include <algorithm>',1)
anchor=b'      run(calls[call_index++], queue, inputs, outputs);'
start=changed.index(b'void run_q24_semantic_fragment(')
assert anchor in changed[start:]
extra=r'''
      for (const auto id : step.outputs) {
        const auto &desc = fixture.description(id);
        std::vector<float> values;
        if (desc.data_type() == c::DataType::kFloat32) values = half_test::read<float>(buffers[id.ordinal()]);
        else if (desc.data_type() == c::DataType::kBFloat16) {
          for (const auto value : half_test::read<BF16>(buffers[id.ordinal()]))
            values.push_back(float(sem::q::decode(value)));
        } else continue;
        const std::string prefix = "C:/public/project-hyle/hyle/.scratch/validation/d5-f40-first-split/";
        std::ofstream file(prefix + "r" + std::to_string(replay_index - 1) + "v" + std::to_string(id.ordinal()) + ".f32", std::ios::binary);
        file.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
        if (replay_index == 1) {
          std::ofstream meta(prefix + "nodes.txt", std::ios::app);
          meta << id.ordinal() << " kind " << int(step.operation.kind()) << " dtype " << int(desc.data_type()) << " inputs";
          for (auto input : step.inputs) meta << " " << input.ordinal();
          meta << " shape";
          for (auto dimension : desc.dimensions()) meta << " " << dimension;
          meta << "\n";
        }
      }
'''
changed=changed[:start]+changed[start:].replace(anchor,anchor+extra.encode(),1)
test=r'''
TEST(CudaMoeFragments, D5GlmOfficialMlaDiagnostic) {
  run_q24_semantic_fragment(d5_glm_remaining_fragments::mla(d5_glm_remaining_fragments::kCases[3]));
}
'''
marker=b'TEST(CudaMoeFragments, D5GlmOfficialMixedRemaining) {'
changed=changed.replace(marker,test.encode()+marker,1)
path.write_bytes(changed)
script=(root/'.scratch/d5-kimi-owner-windows.py').read_text(encoding='utf-8')
script=script.replace('d5-kimi-owner','d5-f40-first-split')
script=script.replace("['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda', 'compute_route_cuda', 'graph_route_cuda']", "['compute_route_cuda']")
script="import os\nos.environ['GTEST_FILTER']='*MoeFragments.D5GlmOfficialMlaDiagnostic'\n"+script
(root/'.scratch/d5-f40-first-split-run.py').write_text(script,encoding='utf-8')
print('Prepared one first-divergence capture.')
