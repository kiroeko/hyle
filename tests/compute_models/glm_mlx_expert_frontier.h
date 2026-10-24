#pragma once

#include <charconv>
#include <chrono>
#include <fstream>
#include <thread>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "glm_mlx_original_roles.h"

// External artifact supply protocol, not a graph/provider/model executor.
// Same explicit section/stage/global_ids/missing_ids frontier as the existing
// DSpark consumer. This code never makes a network request or starts a child.
namespace glm_mlx_expert_frontier {
namespace source=glm_mlx_original_roles;
namespace m=hyle::mem;
inline std::string text(const std::filesystem::path &path) {
  const auto map=source::raw::take(m::FileMapping::open(path.string().c_str()),"map Q32 selected ready receipt");
  if(!map.size()||map.size()>65536)throw std::runtime_error("Q32 selected receipt exceeds metadata bound");
  const auto raw=source::raw::take(map.bytes(),"Q32 selected receipt bytes");
  return {reinterpret_cast<const char*>(raw.data()),raw.size()};
}
inline std::string_view field(std::string_view t,std::string_view key) {
  const auto quoted='"'+std::string(key)+'"';auto first=t.find(quoted);
  if(first==t.npos||t.find(quoted,first+quoted.size())!=t.npos||(first=t.find(':',first+quoted.size()))==t.npos||
      (first=t.find_first_not_of(" \r\n\t",first+1))==t.npos)throw std::runtime_error("Q32 ready lacks unique required field: "+std::string(key));
  return t.substr(first);
}
inline std::string string(std::string_view t,std::string_view key) {
  const auto v=field(t,key);const auto end=v.find('"',1);
  if(v.front()!='"'||end==v.npos||v.substr(1,end-1).find('\\')!=v.npos)throw std::runtime_error("Q32 ready has invalid fixed string");
  return std::string(v.substr(1,end-1));
}
inline std::int64_t number(std::string_view t) {
  const auto end=t.find_first_of(" ,]}\r\n\t");t=t.substr(0,end);std::int64_t n=-1;
  const auto r=std::from_chars(t.data(),t.data()+t.size(),n);
  if(t.empty()||(t.size()>1&&t.front()=='0')||r.ec!=std::errc{}||r.ptr!=t.data()+t.size())throw std::runtime_error("Q32 ready has invalid integer");return n;
}
inline std::optional<std::string> ready_pin(const std::filesystem::path &dir,const source::Data &data,std::string_view section,std::int32_t id) {
  if(!std::filesystem::exists(dir/"ready.json"))return {};
  const auto t=text(dir/"ready.json");const auto done=field(t,"complete");
  if(done.starts_with("false")&&done.size()>5&&std::string_view(" ,}\r\n\t").find(done[5])!=std::string_view::npos)return {};
  if(!done.starts_with("true")||done.size()<5||std::string_view(" ,}\r\n\t").find(done[4])==std::string_view::npos||
      string(t,"repository")!=source::kRepository||string(t,"revision")!=source::kRevision||
      string(t,"section")!=section||number(field(t,"stage"))!=0||
      string(t,"source_plan_sha256")!=data.expert_plan()||!std::filesystem::equivalent(dir,string(t,"directory")))
    throw std::runtime_error("Q32 ready identity/section/source-plan/directory differs");
  const auto ids=field(t,"global_ids");const auto close=ids.find(']');
  if(ids.front()!='['||close==ids.npos)throw std::runtime_error("Q32 ready actual global ID is absent");
  auto contents=ids.substr(1,close-1);const auto first=contents.find_first_not_of(" \r\n\t");
  const auto last=contents.find_last_not_of(" \r\n\t");
  if(first==contents.npos||last==contents.npos||contents.find(',')!=contents.npos)
    throw std::runtime_error("Q32 ready must authenticate exactly the requested original global ID");
  const auto decimal=contents.substr(first,last-first+1);std::int32_t recorded=-1;
  const auto parsed=std::from_chars(decimal.data(),decimal.data()+decimal.size(),recorded);
  if(parsed.ec!=std::errc{}||parsed.ptr!=decimal.data()+decimal.size()||recorded!=id)
    throw std::runtime_error("Q32 ready must authenticate exactly the requested original global ID");
  const auto catalog=field(t,"typed_catalog");const auto end=catalog.find('}');
  if(catalog.front()!='{'||end==catalog.npos)throw std::runtime_error("Q32 ready typed catalogue is absent");
  const auto item=catalog.substr(0,end+1);const auto pin=string(item,"sha256");
  if(string(item,"file")!="catalog.bin"||number(field(item,"bytes"))!=208||!source::pin(pin))
    throw std::runtime_error("Q32 ready needs its actual typed catalogue208B pin");return pin;
}
struct Frontier final {
  std::shared_ptr<source::Data> selected;
  std::string section;
  std::filesystem::path root,request;
  bool wait=false;
  void ensure(std::span<const std::int32_t> globals) {
    const auto expected=selected->prefix()==source::kPrefix?"nextn":selected->prefix()=="model.language_model.layers.43."?"target43":"target44";
    if(section!=expected)throw std::runtime_error("Q32 explicit frontier/source layer differs");
    const auto started=std::chrono::steady_clock::now();bool announced=false;
    for(;;) {
      for(const auto id:selected->missing(globals)) {
        auto decimal=std::to_string(id);if(decimal.size()<3)decimal.insert(0,3-decimal.size(),'0');
        const auto dir=root/section/"stage0"/("expert"+decimal);
        const auto pin=ready_pin(dir,*selected,section,id);
        if(pin)selected->attach_expert(source::ExpertSet::load(dir,*pin,selected->expert_plan(),id,selected->prefix()));
      }
      const auto missing=selected->missing(globals);if(missing.empty())return;
      if(!announced) {
        if(!request.empty()) {
          if(!request.parent_path().empty())std::filesystem::create_directories(request.parent_path());
          auto tmp=request;tmp+=".tmp";std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
          const auto list=[&](auto ids){out<<'[';for(std::size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<']';};
          out<<"{\"section\":\""<<section<<"\",\"stage\":0,\"global_ids\":";
          list(globals);out<<",\"missing_ids\":";list(missing);out<<"}\n";out.close();
          if(!out)throw std::runtime_error("cannot finish Q32 actual selected expert frontier");
#if defined(_WIN32)
          bool replaced=false;
          for(unsigned retry=0;retry<100;++retry) {
            if(MoveFileExW(tmp.c_str(),request.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){replaced=true;break;}
            const auto error=GetLastError();if(error!=ERROR_SHARING_VIOLATION&&error!=ERROR_ACCESS_DENIED)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          if(!replaced)throw std::runtime_error("cannot replace Q32 frontier atomically");
#else
          std::filesystem::rename(tmp,request);
#endif
        }
        std::printf("Q32 actual %s frontier stage0 missing=%zu (%s)\n",section.data(),missing.size(),wait?"waiting for explicit supply":"missing cache fails");std::fflush(stdout);announced=true;
      }
      if(!wait)throw std::runtime_error("Q32 actual selected experts missing; explicit artifact supply is required");
      if(std::chrono::steady_clock::now()-started>std::chrono::minutes(20))throw std::runtime_error("Q32 external expert wait exceeded20 minutes");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
} // namespace glm_mlx_expert_frontier
