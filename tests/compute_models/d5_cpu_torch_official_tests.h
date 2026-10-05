#pragma once

// Included after the ordinary Direct/Graph fixture runner. Identical consumers
// and numerical oracles to the admitted CUDA/Apple programs; CPU uses its
// existing mature router composition.

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_CPU_OFFICIAL_SUITE CpuGraphTorchOfficial
#else
#define HYLE_CPU_OFFICIAL_SUITE CpuTorchOfficial
#endif

TEST(HYLE_CPU_OFFICIAL_SUITE, D5GroupedFp8Batch) {
  d5_grouped_fp8_fragments::projection_batch([](auto fragment) {
    run_q24_semantic_fragment(std::move(fragment));
  });
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5GlmOfficialFp8FullMoe) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    run_q24_semantic_fragment(d5_grouped_fp8_fragments::glm_moe(flash));
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DynamicFp8DenseBatch) {
  namespace fp8 = d5_official_fp8_fragments;
  namespace dynamic = d5_dynamic_fp8_fragments;
  const auto check = [](const fp8::Case &shape) {
    SCOPED_TRACE(shape.profile->model_name);
    SCOPED_TRACE(shape.role);
    for (auto m : {1, 3}) for (bool bf16 : {false, true}) {
      SCOPED_TRACE(m);
      SCOPED_TRACE(bf16);
      run_q24_semantic_fragment(dynamic::projection(shape, m, bf16, false));
      if (bf16) run_q24_semantic_fragment(dynamic::projection(shape, m, true, true));
    }
  };
  for (const auto &shape : fp8::kCases) check(shape);
  check(fp8::kGlmTailCase);
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5GlmOfficialDynamicFp8SharedFfn) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    for (auto m : {1, 3}) {
      SCOPED_TRACE(m);
      run_q24_semantic_fragment(d5_dynamic_fp8_fragments::glm_shared_ffn(m, flash));
    }
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5OfficialFp8DenseNativeBatch) {
  namespace fp8 = d5_official_fp8_fragments;
  for (const auto &shape : fp8::kCases) {
    SCOPED_TRACE(shape.profile->model_name);
    SCOPED_TRACE(shape.role);
    for (auto m : {1, 3}) {
      SCOPED_TRACE(m);
      run_q24_semantic_fragment(fp8::projection(shape, m));
    }
  }
  for (auto m : {1, 3}) {
    SCOPED_TRACE("GLM partial N block");
    SCOPED_TRACE(m);
    run_q24_semantic_fragment(fp8::projection(fp8::kGlmTailCase, m));
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5GlmOfficialMixedRemaining) {
  namespace glm = d5_glm_remaining_fragments;
  run_q24_semantic_fragment(glm::official_flash_clamped_activation());
  for (const auto &model : glm::kCases) {
    if (!model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    run_q24_semantic_fragment(glm::dense_ffn(model));
    for (int tail : {0, 3}) {
      if (!model.flash && tail) continue;
      run_q24_semantic_fragment(glm::indexer(model, tail));
      run_q24_semantic_fragment(glm::sparse_attention(model, tail));
    }
    glm::check_index_carry(model, run_q24_semantic_fragment);
    if (model.flash) {
      run_q24_semantic_fragment(glm::mla(model));
      glm::check_mtp_head(model, run_q24_semantic_fragment);
    }
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekOfficialCompression) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    const auto &model = d5_moe_execution_fragments::cases::kCases[index];
    for (std::int64_t ratio : {4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::compressor(model, ratio));
    }
    run_q24_semantic_fragment(d5_deepseek_official_attention::compressor(model, 4, true));
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekOfficialAttention) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    const auto &model = d5_moe_execution_fragments::cases::kCases[index];
    for (std::int64_t ratio : {0, 4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::attention(model, ratio));
    }
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekOfficialMtpAndFullHead) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    d5_deepseek_official_attention::check_mtp_head(
        d5_moe_execution_fragments::cases::kCases[index], run_q24_semantic_fragment, false);
  }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekOriginalFp4Projections) {
  for (int index : {3, 4})
    for (bool down : {false, true})
      for (bool shared : {false, true})
        for (std::int64_t tokens : {1, 3}) {
          SCOPED_TRACE(index);
          SCOPED_TRACE(down);
          SCOPED_TRACE(shared);
          SCOPED_TRACE(tokens);
          run_q24_semantic_fragment(d5_deepseek_official_fragments::projection(
              d5_moe_execution_fragments::cases::kCases[index], down, shared, tokens));
        }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekOfficialMoe) {
  for (int index : {3, 4})
    for (bool hash : {false, true}) {
      SCOPED_TRACE(index);
      SCOPED_TRACE(hash);
      run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(
          d5_moe_execution_fragments::cases::kCases[index], hash, nullptr, false));
    }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, D5DeepSeekExpertActivationRoundtrip) {
  for (std::int64_t width : {7168, 4096})
    for (std::int64_t rows : {1, 3, 33}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(rows);
      run_q24_semantic_fragment(d5_quantization_roundtrip::fragment(false, rows, width, 128));
    }
}

TEST(HYLE_CPU_OFFICIAL_SUITE, OriginalFp8WeightDecodeAllFiniteCodes) {
  for (bool bf16 : {false, true})
    run_q24_semantic_fragment(d5_fp8_weight_preparation::decode_weights(129, 256, bf16));
}
#undef HYLE_CPU_OFFICIAL_SUITE
