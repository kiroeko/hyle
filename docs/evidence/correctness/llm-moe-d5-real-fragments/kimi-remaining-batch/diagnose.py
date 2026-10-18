from pathlib import Path

root = Path.cwd()
cpu = root / 'tests/compute_route_cpu/d5_qwen38_official_test.cc'
backup = root / '.scratch/d5-kimi-cpu-pre-diagnostic.cc'
assert not backup.exists()
backup.write_bytes(cpu.read_bytes())
data = cpu.read_bytes()
data = data.replace(b'#include <cstddef>', b'#include <cstdio>\n#include <cstddef>', 1)
needle = b'      feedback.observe(output, actual);'
insert = '''      if (fixture.region == 0x02030701 || fixture.region == 0x070701) {
        int printed = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const auto a = std::bit_cast<std::uint16_t>(sem::q::encode(actual[i]));
          const auto e = std::bit_cast<std::uint16_t>(sem::q::encode(expected.outputs[output][i]));
          if (a != e && printed++ < 6)
            std::printf("BF16_DIAG profile=%.*s region=%llu output=%zu index=%zu actual=%.12g expected=%.12g bits=%04x/%04x\\n",
                int(fixture.profile->variant.size()), fixture.profile->variant.data(),
                static_cast<unsigned long long>(fixture.region), output, i,
                double(actual[i]), double(expected.outputs[output][i]), unsigned(a), unsigned(e));
        }
      }
'''.encode()
assert data.count(needle) == 1
cpu.write_bytes(data.replace(needle, insert + needle))

path = root / 'tests/compute_models/d5_kimi_remaining_fragments.h'
data = path.read_bytes()
# Additional observations do not change the operations or data under test.
data = data.replace(b'vb.output, gated, out.output};', b'vb.output, gated, out.output, normalized, gate.output};')
data = data.replace(b'projected, gv, result};', b'projected, gv, result, nx, gate_values};')
# Original res_push appends raw input. The fixed-capacity consumer retains
# unchanged padded rows using checked Gather; no assignment-Scatter kernel.
data = data.replace(b'auto slot = f.input(dense({1}, c::DataType::kInt32));',
                    b'auto bank_rows = f.input(dense({8}, c::DataType::kInt32));')
data = data.replace(b'? f.one(c::Operation(c::Scatter{.axis = 0}), {bank, slot, current})',
                    b'? f.one(c::Operation(c::Gather{}),\n              {f.one(c::Operation(c::Concat{.axis = 0}), {bank, current}), bank_rows})')
data = data.replace(b'    sem::Replay replay{{bytes(bank_values),',
                    b'    std::vector<std::int32_t> rows(8);\n    std::iota(rows.begin(), rows.end(), 0);\n    if (boundary) rows[step] = 8;\n    sem::Replay replay{{bytes(bank_values),')
data = data.replace(b'bytes(std::vector<std::int32_t>{step})', b'bytes(rows)')
path.write_bytes(data)

driver = (root / '.scratch/d5-kimi-windows.py').read_text(encoding='utf-8')
driver = driver.replace('.scratch/validation/d5-kimi-remaining', '.scratch/validation/d5-kimi-diagnostic')
driver = driver.replace('exec(compile(source,', "source = source.replace(\"for entry in ('compute', 'graph'):\", \"for entry in ('compute',):\")\nexec(compile(source,")
(root / '.scratch/d5-kimi-diagnostic-windows.py').write_text(driver, encoding='utf-8')
