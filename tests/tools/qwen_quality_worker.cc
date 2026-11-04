#include "../../engine/models/qwen35/main.cc"
#include "inference_quality_protocol.h"
#include <sstream>

int main(int argc,char **argv) {
  namespace q=qwen35;namespace wire=inference_quality;
  try {
    if (argc==2 && std::string_view(argv[1])=="--help") {
      std::cout<<"qwen_quality_worker INPUT CAPACITY MAX_OUTPUT_ROWS\nBinary HYQL0001 stream;stdin P/D/R/X commands.\n";return 0;
    }
    if (argc!=4) throw std::runtime_error("qwen_quality_worker INPUT CAPACITY MAX_OUTPUT_ROWS");
    wire::binary_stdout();
    auto source=q::prepare_weight_source(std::filesystem::path(argv[1])/"weights.tsv",false);
    const auto geometry=source.roles.geometry;
    const auto precision=q::PrecisionProfile::from_roles(source.roles);
    const auto capacity=std::stoull(argv[2]), maximum=std::stoull(argv[3]);
    if (!maximum || maximum>1000000) throw std::runtime_error("invalid row bound");
    const auto plan=q::cache::plan(geometry,source.cache_row_bytes,capacity,1,capacity,
        q::recurrent_bytes(q::state_schema_for(geometry,capacity,precision),geometry),
        1,1,false,true,1,0,precision);
    q::Model model(std::move(source),geometry,capacity,plan,{},false,q::ReplayPolicy::kAuto,
        SIZE_MAX,{},0,{.primary=q::BackendKind::kCuda});
    const std::array rows{std::size_t{1}}, chunks{std::size_t{512}};
    const std::array<std::size_t,6> tiers{16,32,64,128,256,512};
    model.prepare_family(rows);
    if (!model.prepare_prefill_family(tiers)) throw std::runtime_error("prefill family missing");
    model.prepare_family(chunks,false,1,q::GraphRole::kText,false,q::TextCompletion::kStateOnly);
    if (!model.prepare_prefill_family(chunks,q::TextCompletion::kStateOnly))
      throw std::runtime_error("state family missing");
    const auto uses=maximum+2048;
    model.plan_replay_uses(1,1,uses);
    for (auto tier:tiers) model.plan_replay_uses(tier,1,uses,q::GraphRole::kText,
        q::TextCompletion::kEmitNext,0,q::RowPacking::kPrefill);
    model.plan_replay_uses(512,1,uses,q::GraphRole::kText,q::TextCompletion::kStateOnly);
    model.plan_replay_uses(512,1,uses,q::GraphRole::kText,q::TextCompletion::kStateOnly,0,q::RowPacking::kPrefill);
    model.prepare_shared_execution_workspace();model.prepare_execution(rows);
    model.prepare_execution(chunks,1,q::GraphRole::kText,q::TextCompletion::kStateOnly);
    const auto cold=q::take(q::prepare_cold_prefill(model,true,false,0));
    if (!cold.retired || model.snapshot_versions.live) throw std::runtime_error("cold state not retired");
    std::shared_ptr<q::Session> session;
    std::uint64_t case_id=0,position=0,total=0;
    const auto retire=[&] {
      if (session) {
        q::check(session->stop());q::check(session->drain().status);
        const auto done=session->try_retire();q::check(done.status);
        if (!done.retired) throw std::runtime_error("session still live");
        session.reset();
      }
      if (model.snapshot_versions.live || model.active_step || model.carrier_borrowed ||
          model.page_inventory.free_ids.size()!=plan.max_pages)
        throw std::runtime_error("original resources not retired");
    };
    const auto evaluate=[&](std::span<const std::int32_t> ids,bool packed) {
      if (++total>maximum) throw std::runtime_error("row bound exceeded");
      if (packed) {
        const std::array requests{q::EvaluationRequest{session,ids}};
        model.prepare_prefill_evaluations(requests,nullptr,q::GraphRole::kText,q::TextCompletion::kEmitNext);
      } else model.prepare_evaluation(*session,ids,nullptr,q::GraphRole::kText,q::TextCompletion::kEmitNext);
      std::vector<float> logits;
      std::int32_t next=-1;
      try {
        model.submit_evaluation(*session);
        const std::array outputs{q::EvaluationOutput{.session=session,.capture_tensors=true}};
        model.finish_evaluations(outputs,std::span(&next,1),[&](std::size_t member,const q::CompletedEvaluationView &view) {
          if (member || view.requests!=1 || view.logits.description()!=q::dense(q::Type::kFloat32,{1,model.vocabulary}))
            throw std::runtime_error("quality output shape differs");
          logits.resize(std::size_t(model.vocabulary));
          q::check(model.execution.read(view.logits.storage(),std::as_writable_bytes(std::span(logits))));
        });
      } catch (...) { (void)model.abort_episode();throw; }
      if (model.active_step || model.carrier_borrowed || session->pending || session->head->sampling.draw ||
          wire::top1(logits)!=next) throw std::runtime_error("quality retirement/RNG/greedy mismatch");
      wire::emit(2,std::uint32_t(model.vocabulary),case_id,position,session->length(),0,logits);
    };
    wire::emit(1,std::uint32_t(model.vocabulary));
    std::string line;
    while (std::getline(std::cin,line)) {
      std::istringstream input(line);char command=0;input>>command;
      if (command=='P') {
        retire();std::size_t count=0;input>>case_id>>count;
        if (!input || !count || count>512 || count>=capacity) throw std::runtime_error("invalid prefill command");
        std::vector<std::int32_t> ids(count);
        for (auto &id:ids) if (!(input>>id) || id<0 || id>=model.vocabulary) throw std::runtime_error("invalid prompt token");
        std::string extra;if (input>>extra) throw std::runtime_error("extra prefill field");
        position=0;const std::array initial{count};session=model.create_session(initial);
        evaluate(ids,true);
      } else if (command=='D') {
        std::uint64_t expected=0;std::int32_t token=-1;input>>expected>>token;
        if (!input || !session || expected!=case_id || token<0 || token>=model.vocabulary)
          throw std::runtime_error("invalid decode command");
        std::string extra;if (input>>extra) throw std::runtime_error("extra decode field");
        ++position;evaluate(std::span(&token,1),false);
      } else if (command=='R') {
        retire();wire::emit(3,std::uint32_t(model.vocabulary));
      } else if (command=='X') {
        retire();wire::emit(4,std::uint32_t(model.vocabulary));return 0;
      } else throw std::runtime_error("unknown quality command");
    }
    retire();throw std::runtime_error("quality input ended without clean exit");
  } catch (const std::exception &e) {std::cerr<<"quality worker: "<<e.what()<<'\n';return 1;}
}
