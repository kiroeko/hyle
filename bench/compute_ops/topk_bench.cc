#include "topk_bench.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nanobench.h>
#include "perf_process.h"

namespace {
namespace nb=ankerl::nanobench;
std::string environment(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t size=0;
  if (_dupenv_s(&value,&size,name)!=0) throw std::runtime_error("cannot read benchmark environment");
  const std::string result=value?value:"";std::free(value);return result;
#else
  const auto *value=std::getenv(name);return value?value:"";
#endif
}
bool enabled(const char *name) {const auto value=environment(name);return !value.empty()&&value!="0";}

void measure(int rows,int count) {
  g08::Data data(rows,count);
  const auto aa=environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty()&&aa!="native"&&aa!="selected") throw std::runtime_error("invalid TopK A/A route");
  const auto construct=[&](bool original){return original?g08::make_reference(data):g08::make_candidate(data);};
  std::unique_ptr<g08::Arm> candidate,peer;
  if (enabled("HYLE_BENCH_PEER_ALLOCATION_FIRST")) {
    peer=construct(aa!="selected");candidate=construct(aa=="native");
  } else {candidate=construct(aa=="native");peer=construct(aa!="selected");}
  candidate->verify();peer->verify();
  for (int cycle=0;cycle<8;++cycle) {candidate->execute();peer->execute();}
  std::fprintf(stderr,"# G08_ORACLE route=%s R=%d V=%d K=%d stored_bits_and_first_index=PASS cycles=3 warm_cycles=8 readback_bytes=%d\n",
    HYLE_G08_ROUTE_NAME,rows,g08::kVocabulary,count,rows*count*8);
  if (!enabled("HYLE_BENCH_VERIFY_ONLY")) {
    nb::Bench bench;
    bench.title(std::string(HYLE_G08_ROUTE_NAME)+" TopK R="+std::to_string(rows)+" V=248320 K="+std::to_string(count))
      .batch(1).epochs(HYLE_G08_PAIRED_ROUNDS).minEpochIterations(1)
      .minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
    const auto a=[&]{candidate->execute();};const auto b=[&]{peer->execute();};
    if (enabled("HYLE_BENCH_RIVAL_FIRST"))
      (void)bench.compare("Original85 TopK",b,"HYLE TopK",a);
    else (void)bench.compare("HYLE TopK",a,"Original85 TopK",b);
  }
  candidate->finish();peer->finish();
}
} // namespace

int main(int,char **) {
  try {
    const auto qos=hyle::bench::perf::configure_thread_qos();
    std::fprintf(stderr,"# HYLE_TOPK_RUN run_nonce=%s apple_qos=%s qos_ok=%s route=%s verify_only=%d paired_rounds=%d placement=%d execution_order=%s allocation_order=%s aa_route=%s\n",
      hyle::bench::perf::run_nonce(),qos.name,qos.ok,HYLE_G08_ROUTE_NAME,enabled("HYLE_BENCH_VERIFY_ONLY"),
      HYLE_G08_PAIRED_ROUNDS,HYLE_G08_PLACEMENT,enabled("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",
      enabled("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",environment("HYLE_BENCH_AA_ROUTE").c_str());
    measure(1,1);measure(16,1);
#if HYLE_G08_APPLE
    measure(16,3);
#endif
    return 0;
  } catch (const std::exception &error) {std::fprintf(stderr,"FAIL G08 TopK: %s\n",error.what());return 1;}
}
