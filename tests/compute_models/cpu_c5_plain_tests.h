#pragma once
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define HYLE_C5_PLAIN_SUITE CpuGraphC5Plain
#else
#define HYLE_C5_PLAIN_SUITE CpuC5Plain
#endif
class HYLE_C5_PLAIN_SUITE : public ::testing::TestWithParam<const cpu_c5_plain_inputs::Case *> {};
TEST_P(HYLE_C5_PLAIN_SUITE, ActualUncompressedTextRolesPreserveStorageAndReplay) {
  const auto &item=*GetParam();const auto identity=cpu_c5_plain_fixture::profile(item);
  SCOPED_TRACE(item.id);SCOPED_TRACE(identity.artifact_repository);
  const std::string_view kind=item.kind;namespace p=cpu_c5_plain_fixture;
  if(kind=="projection"||kind=="head") {
    const auto rows=kind=="head"?4096:item.shape[0];
    for(std::int64_t first=0;first<item.shape[0];first+=rows) {
      run_q24_semantic_fragment(p::projection(item,&identity,std::min<std::int64_t>(rows,item.shape[0]-first),first));
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  } else if(kind=="embedding"||kind=="position_lookup"||kind=="integer_lookup")run_q24_semantic_fragment(p::lookup(item,&identity));
  else if(kind=="norm"||kind=="norm_zero_centered")run_q24_semantic_fragment(p::norm(item,&identity));
  else if(kind=="conv")run_q24_semantic_fragment(p::convolution(item,&identity));
  else run_q24_semantic_fragment(p::vector_parameter(item,&identity));
}
INSTANTIATE_TEST_SUITE_P(C5ActualPlainContracts,HYLE_C5_PLAIN_SUITE,
    ::testing::ValuesIn([]{std::vector<const cpu_c5_plain_inputs::Case *> result;
      for(const auto &item:cpu_c5_plain_inputs::cases)result.push_back(&item);return result;}()),
    [](const auto &parameter){return std::string(parameter.param->id);});
#undef HYLE_C5_PLAIN_SUITE
