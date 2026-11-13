#include "../../engine/models/qwen35/main.cc"
#include "inference_quality_protocol.h"
#include <sstream>

int main(int argc,char **argv) {
  namespace q=qwen35;namespace wire=inference_quality;
  try {
    if (argc==2 && std::string_view(argv[1])=="--help") {
      std::cout<<"qwen_quality_worker INPUT CAPACITY MAX_OUTPUT_ROWS [--mtp [1|3] [--mtp-draft-vocabulary FILE]]\nBinary HYQL0001 stream;stdin P/D/R/X commands.\n"
          "--mtp evaluates the first verifier row of a real bounded round and publishes its one-token prefix.\n";return 0;
    }
    const bool mtp=argc>=5 && std::string_view(argv[4])=="--mtp";
    if (argc!=4 && !mtp) throw std::runtime_error("qwen_quality_worker INPUT CAPACITY MAX_OUTPUT_ROWS [--mtp [1|3] [--mtp-draft-vocabulary FILE]]");
    int option=mtp?5:argc;
    const auto draft_option=option<argc && std::string_view(argv[option])!="--mtp-draft-vocabulary"
        ?std::string_view(argv[option++]):std::string_view("1");
    if(draft_option!="1" && draft_option!="3")throw std::runtime_error("quality MTP supports fixed K1 or K3");
    const auto drafts=draft_option=="3"?std::size_t{3}:std::size_t{1};
    std::filesystem::path draft_vocabulary;
    if(option<argc){
      if(std::string_view(argv[option++])!="--mtp-draft-vocabulary" || option>=argc || !*argv[option])
        throw std::runtime_error("quality draft vocabulary requires its existing source-bound file");
      draft_vocabulary=argv[option++];
    }
    if(option!=argc)throw std::runtime_error("extra quality worker options");
    wire::binary_stdout();
    auto source=q::prepare_weight_source(std::filesystem::path(argv[1])/"weights.tsv",mtp,draft_vocabulary);
    const auto draft_vocabulary_rows=source.draft_vocabulary?source.draft_vocabulary->size():0;
    const auto geometry=source.roles.geometry;
    const auto precision=q::PrecisionProfile::from_roles(source.roles);
    const auto capacity=std::stoull(argv[2]), maximum=std::stoull(argv[3]);
    if (!maximum || maximum>1000000) throw std::runtime_error("invalid row bound");
    const auto plan=q::cache::plan(geometry,source.cache_row_bytes,capacity,1,capacity,
        q::recurrent_bytes(q::state_schema_for(geometry,capacity,precision),geometry),
        1,mtp?drafts+1:1,mtp,true,mtp?3:1,mtp&&drafts>1?2:0,precision,mtp&&drafts>1?drafts:0);
    q::Model model(std::move(source),geometry,capacity,plan,{},false,q::ReplayPolicy::kAuto,
        SIZE_MAX,{},0,{.primary=q::BackendKind::kCuda});
    if(mtp){model.configure_draft_policy({drafts});model.prepare_default_device_mtp_successor(true);}
    const auto single_role=mtp?q::GraphRole::kMtpSingle:q::GraphRole::kText;
    const auto prefill_role=mtp?q::GraphRole::kMtpPrefill:q::GraphRole::kText;
    const std::array rows{std::size_t{1}}, chunks{std::size_t{512}};
    const std::array<std::size_t,6> tiers{16,32,64,128,256,512};
    model.prepare_family(rows,false,1,single_role);
    if (!model.prepare_prefill_family(tiers,q::TextCompletion::kEmitNext,0,1,prefill_role)) throw std::runtime_error("prefill family missing");
    model.prepare_family(chunks,false,1,prefill_role,false,q::TextCompletion::kStateOnly);
    if (!model.prepare_prefill_family(chunks,q::TextCompletion::kStateOnly,0,1,prefill_role))
      throw std::runtime_error("state family missing");
    const auto uses=maximum+2048;
    model.plan_replay_uses(1,1,uses,single_role);
    for (auto tier:tiers) model.plan_replay_uses(tier,1,uses,prefill_role,
        q::TextCompletion::kEmitNext,0,q::RowPacking::kPrefill);
    model.plan_replay_uses(512,1,uses,prefill_role,q::TextCompletion::kStateOnly);
    model.plan_replay_uses(512,1,uses,prefill_role,q::TextCompletion::kStateOnly,0,q::RowPacking::kPrefill);
    if(mtp){
      const std::array round_rows{std::size_t{drafts+1}};
      for(const auto role:{q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound}){
        model.prepare_family(round_rows,false,1,role);model.plan_replay_uses(drafts+1,1,uses,role);
      }
      if(drafts>1)model.prepare_recursive_round_programs(0,false,uses,drafts-1);
    }
    model.prepare_execution_storage();model.prepare_execution(rows,1,single_role);
    model.prepare_execution(chunks,1,prefill_role,q::TextCompletion::kStateOnly);
    if(mtp){const std::array round_rows{std::size_t{drafts+1}};
      model.prepare_execution(round_rows,1,q::GraphRole::kMtpRound);
      model.prepare_execution(round_rows,1,q::GraphRole::kMtpCatchupRound);}
    const auto cold=q::take(q::prepare_cold_prefill(model,true,mtp,0));
    if (!cold.retired || model.snapshot_versions.live) throw std::runtime_error("cold state not retired");
    std::shared_ptr<q::Session> session;
    std::uint64_t case_id=0,position=0,total=0,verifier_rows=0;
    const auto retire=[&] {
      if (session) {
        q::check(session->stop());q::check(session->drain().status);
        const auto done=session->try_retire();q::check(done.status);
        if (!done.retired) throw std::runtime_error("session still live");
        session.reset();
      }
      if (model.snapshot_versions.live || model.active_step || model.carrier_borrowed ||
          model.page_inventory.free_ids.size()!=plan.max_pages ||
          (mtp && model.draft_inventory.free_ids.size()!=plan.max_pages))
        throw std::runtime_error("original resources not retired");
    };
    const auto evaluate=[&](std::span<const std::int32_t> ids,bool packed) {
      if (++total>maximum) throw std::runtime_error("row bound exceeded");
      if (packed) {
        const std::array requests{q::EvaluationRequest{session,ids}};
        model.prepare_prefill_evaluations(requests,nullptr,prefill_role,q::TextCompletion::kEmitNext);
      } else model.prepare_evaluation(*session,ids,nullptr,mtp
          ?(session->head->draft->lag?q::GraphRole::kMtpCatchupRound:q::GraphRole::kMtpRound)
          :q::GraphRole::kText,q::TextCompletion::kEmitNext);
      if(mtp&&!packed&&drafts>1&&model.active_step->phase->draft_chain_steps!=drafts)
        throw std::runtime_error("quality missed the complete device draft chain");
      std::vector<float> logits;
      std::int32_t next=-1;
      try {
        model.submit_evaluation(*session);
        const std::array outputs{q::EvaluationOutput{.session=session,.capture_tensors=true}};
        const bool verifier=mtp&&!packed;
        const auto observe=[&](std::size_t member,const q::CompletedEvaluationView &view) {
          if (member || view.requests!=1 || view.logits.description()!=q::dense(q::Type::kFloat32,{static_cast<std::int64_t>(verifier?drafts+1:1),model.vocabulary}))
            throw std::runtime_error("quality output shape differs");
          logits.resize(std::size_t(model.vocabulary)*(verifier?drafts+1:1));
          q::check(model.execution.read(view.logits.storage(),std::as_writable_bytes(std::span(logits))));
          if(verifier)logits.resize(std::size_t(model.vocabulary));
        };
        if(verifier){
          // The controller supplies the BF16 teacher's next pending input.
          // Commit exactly that input through the original accepted-prefix
          // machinery; the speculative suffix never enters its prefix.
          model.finish_selected_evaluations(outputs,std::span(&next,1),
              [&](std::span<const std::int32_t> selected,q::StepState &step,std::span<std::int32_t> next){
                if(selected.size()!=drafts+1 || step.input_tokens!=drafts+1)throw std::runtime_error("quality missed real bounded verifier");
                step.members[0].accepted_tokens=1;next[0]=selected[0];
              },[&](q::StepState &step){if(drafts>1)model.repair_recursive_draft(step);},observe);
          ++verifier_rows;
        }else model.finish_evaluations(outputs,std::span(&next,1),observe);
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
        if (!input || !count || count>=capacity) throw std::runtime_error("invalid prefill command");
        std::vector<std::int32_t> ids(count);
        for (auto &id:ids) if (!(input>>id) || id<0 || id>=model.vocabulary) throw std::runtime_error("invalid prompt token");
        std::string extra;if (input>>extra) throw std::runtime_error("extra prefill field");
        position=0;
        const std::array initial{(std::min)(count,std::size_t{512})};
        session=model.create_session(initial,{},prefill_role,{},false,
            count>512?q::TextCompletion::kStateOnly:q::TextCompletion::kEmitNext);
        std::size_t consumed=0;
        while (count-consumed>512) {
          const std::span chunk(ids.data()+consumed,std::size_t{512});
          const std::array requests{q::EvaluationRequest{session,chunk}};
          model.prepare_prefill_evaluations(requests,nullptr,prefill_role,q::TextCompletion::kStateOnly);
          try {
            model.submit_evaluation(*session);
            (void)q::finish_evaluation(model,*session,"quality_prefill_chunk");
          } catch (...) { (void)model.abort_episode();throw; }
          consumed+=chunk.size();
          if (session->length()!=consumed || model.active_step || model.carrier_borrowed || session->pending ||
              session->head->sampling.draw)
            throw std::runtime_error("quality prefill chunk retirement/RNG mismatch");
        }
        evaluate(std::span(ids.data()+consumed,count-consumed),true);
      } else if (command=='D') {
        std::uint64_t expected=0;std::int32_t token=-1;input>>expected>>token;
        if (!input || !session || expected!=case_id || token<0 || token>=model.vocabulary)
          throw std::runtime_error("invalid decode command");
        std::string extra;if (input>>extra) throw std::runtime_error("extra decode field");
        ++position;evaluate(std::span(&token,1),false);
      } else if (command=='R') {
        retire();wire::emit(3,std::uint32_t(model.vocabulary));
      } else if (command=='X') {
        retire();std::cerr<<"quality_verified_rows="<<verifier_rows<<" total_positions="<<total<<" mtp="<<mtp<<" max_drafts="<<drafts
            <<" draft_vocabulary_rows="<<draft_vocabulary_rows<<" draft_head_rows="
            <<(model.resources_->draft_head?model.resources_->draft_head->projection_weight().description.dimensions()[0]:0)<<'\n';
        wire::emit(4,std::uint32_t(model.vocabulary));return 0;
      } else throw std::runtime_error("unknown quality command");
    }
    retire();throw std::runtime_error("quality input ended without clean exit");
  } catch (const std::exception &e) {std::cerr<<"quality worker: "<<e.what()<<'\n';return 1;}
}
