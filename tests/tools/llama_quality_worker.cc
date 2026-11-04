#include "llama.h"
#include "ggml-backend.h"
#include "inference_quality_protocol.h"
#include <array>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif

int main(int argc,char **argv) {
  namespace wire=inference_quality;
  llama_model *model=nullptr;llama_context *ctx=nullptr;
  try {
    if (argc==2 && std::string_view(argv[1])=="--help") {
      std::cout<<"llama_quality_worker OFFICIAL GGUF CAPACITY\nBinary HYQL0001 stream;stdin T/P/D/R/X commands.\n";return 0;
    }
    if (argc!=4) throw std::runtime_error("llama_quality_worker OFFICIAL GGUF CAPACITY");
    wire::binary_stdout();
    ggml_backend_load_all_from_path(argv[1]);llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=99;
    model=llama_model_load_from_file(argv[2],mp);
    if (!model) throw std::runtime_error("model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=std::stoul(argv[3]);
    cp.n_batch=cp.n_ubatch=512;cp.n_seq_max=1;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=cp.type_v=GGML_TYPE_BF16;
    ctx=llama_init_from_model(model,cp);
    if (!ctx || llama_n_ctx(ctx)!=cp.n_ctx) throw std::runtime_error("effective context differs");
#if defined(_WIN32)
    for (const wchar_t *name:{L"llama.dll",L"ggml-cuda.dll",L"ggml-base.dll"}) {
      wchar_t path[32768]{};
      if (!GetModuleFileNameW(GetModuleHandleW(name),path,32768) ||
          !std::filesystem::equivalent(path,std::filesystem::path(argv[1])/name))
        throw std::runtime_error("official runtime DLL identity differs");
      std::wcerr<<L"runtime "<<name<<L"="<<path<<L'\n';
    }
#endif
    const auto *vocab=llama_model_get_vocab(model);const auto size=llama_vocab_n_tokens(vocab);
    std::uint64_t case_id=0,position=0,length=0;bool active=false;
    const auto evaluate=[&](std::vector<llama_token> &ids) {
      if (ids.empty() || ids.size()>512 || length+ids.size()>cp.n_ctx)
        throw std::runtime_error("invalid decode size");
      if (llama_decode(ctx,llama_batch_get_one(ids.data(),int32_t(ids.size()))))
        throw std::runtime_error("llama_decode failed");
      llama_synchronize(ctx);length+=ids.size();
      const auto *logits=llama_get_logits_ith(ctx,-1);
      if (!logits) throw std::runtime_error("missing logits");
      const auto row=std::span(logits,std::size_t(size));const auto best=wire::top1(row);
      wire::emit(2,std::uint32_t(size),case_id,position,length,0,row,
          2U|(llama_vocab_is_eog(vocab,best)?1U:0U));
    };
    wire::emit(1,std::uint32_t(size));
    std::string line;
    while (std::getline(std::cin,line)) {
      std::istringstream input(line);char command=0;input>>command;
      if (command=='P') {
        std::size_t count=0;input>>case_id>>count;
        if (!input || !count || count>512) throw std::runtime_error("invalid prefill command");
        std::vector<llama_token> ids(count);
        for (auto &id:ids) if (!(input>>id) || id<0 || id>=size) throw std::runtime_error("invalid token");
        std::string extra;if (input>>extra) throw std::runtime_error("extra prefill field");
        llama_memory_clear(llama_get_memory(ctx),true);length=position=0;active=true;evaluate(ids);
      } else if (command=='D') {
        std::uint64_t expected=0;llama_token token=-1;input>>expected>>token;
        if (!input || !active || expected!=case_id || token<0 || token>=size)
          throw std::runtime_error("invalid decode command");
        std::string extra;if (input>>extra) throw std::runtime_error("extra decode field");
        ++position;std::vector<llama_token> ids{token};evaluate(ids);
      } else if (command=='T') {
        std::uint64_t id=0;std::string hex;input>>id>>hex;
        if (!input || hex.empty() || hex.size()%2 || hex.size()>131072)
          throw std::runtime_error("invalid tokenizer command");
        const auto nibble=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("invalid hex");};
        std::string user;
        for (std::size_t i=0;i<hex.size();i+=2) user.push_back(char(nibble(hex[i])*16+nibble(hex[i+1])));
        const std::array messages{llama_chat_message{"system","You are a helpful assistant. Answer directly and follow the requested format."},
            llama_chat_message{"user",user.c_str()}};
        const char *chat=llama_model_chat_template(model,nullptr);
        std::vector<char> text(65536);
        const int chars=llama_chat_apply_template(chat,messages.data(),messages.size(),true,text.data(),int(text.size()));
        if (chars<=0 || chars>int(text.size())) throw std::runtime_error("unsupported or oversized chat template");
        const int need=llama_tokenize(vocab,text.data(),chars,nullptr,0,true,true);
        if (need>=0 || need< -512) throw std::runtime_error("prompt exceeds512 tokens");
        std::vector<llama_token> tokens(std::size_t(-need));
        if (llama_tokenize(vocab,text.data(),chars,tokens.data(),int(tokens.size()),true,true)!=int(tokens.size()))
          throw std::runtime_error("tokenization changed");
        wire::emit(5,std::uint32_t(size),id,0,tokens.size());
        std::cout.write(reinterpret_cast<const char *>(tokens.data()),std::streamsize(tokens.size()*sizeof(llama_token)));
        std::cout.flush();if (!std::cout) throw std::runtime_error("token stream write failed");
      } else if (command=='R') {
        llama_memory_clear(llama_get_memory(ctx),true);active=false;length=position=0;
        wire::emit(3,std::uint32_t(size));
      } else if (command=='X') {
        llama_synchronize(ctx);llama_free(ctx);ctx=nullptr;llama_model_free(model);model=nullptr;
        llama_backend_free();wire::emit(4,std::uint32_t(size));return 0;
      } else throw std::runtime_error("unknown quality command");
    }
    throw std::runtime_error("quality input ended without clean exit");
  } catch (const std::exception &e) {
    std::cerr<<"quality peer: "<<e.what()<<'\n';
    if (ctx)llama_free(ctx);if(model)llama_model_free(model);llama_backend_free();return 1;
  }
}
