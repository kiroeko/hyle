#pragma once

#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "glm_original_moe_roles.h"

// Explicit external supply handshake, never lookup-triggered HTTP. The tiny
// ready/catalog metadata is emitted only after the existing bounded downloader
// authenticated all six original planes; ExpertSet hashes/maps those bytes.
namespace glm_original_expert_frontier {
namespace source=glm_original_moe_roles;
namespace m=hyle::mem;
inline bool pin(std::string_view value) {
  return value.size()==64&&std::ranges::all_of(value,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
inline std::string text(const std::filesystem::path &path,std::size_t maximum) {
  const auto mapping=source::shared::take(m::FileMapping::open(path.string().c_str()),"map explicit original GLM supply metadata");
  if(!mapping.size()||mapping.size()>maximum)throw std::runtime_error("original GLM supply metadata extent differs");
  const auto bytes=source::shared::take(mapping.bytes(),"mapped original GLM supply metadata");
  return {reinterpret_cast<const char*>(bytes.data()),bytes.size()};
}
inline std::vector<std::string_view> split(std::string_view input,char separator) {
  std::vector<std::string_view> result;
  for(;;) {const auto at=input.find(separator);result.push_back(input.substr(0,at));if(at==input.npos)break;input.remove_prefix(at+1);}
  return result;
}
inline std::size_t decimal(std::string_view input) {
  std::size_t value=0;const auto parsed=std::from_chars(input.data(),input.data()+input.size(),value);
  if(input.empty()||(input.size()>1&&input.front()=='0')||parsed.ec!=std::errc{}||parsed.ptr!=input.data()+input.size())
    throw std::runtime_error("original GLM supply metadata decimal differs");
  return value;
}
struct Contract final {source::SourceIdentity identity;std::string plan_sha256;source::WeightStorage storage;std::string section;};
inline std::shared_ptr<const source::ExpertSet> load_ready(const std::filesystem::path &directory,
    const Contract &contract,std::int32_t global,const source::Geometry &model) {
  if(!std::filesystem::exists(directory/"ready.tsv"))return {};
  const auto ready=text(directory/"ready.tsv",2048);
  if(ready.back()!='\n'||ready.find('\r')!=ready.npos)throw std::runtime_error("original GLM ready needs exact LF metadata");
  const auto fields=split(std::string_view(ready).substr(0,ready.size()-1),'\t');
  if(fields.size()!=9||fields[0]!="HYGLMR01"||fields[1]!="1"||fields[2]!=contract.identity.repository||
      fields[3]!=contract.identity.revision||fields[4]!=contract.identity.layer_prefix||fields[5]!=contract.identity.scale_member||
      fields[6]!=contract.plan_sha256||decimal(fields[7])!=std::size_t(global)||!pin(fields[8]))
    throw std::runtime_error("original GLM ready artifact/layer/plan/global/catalog identity differs");
  const auto catalog=text(directory/"catalog.tsv",16384);
  if(source::shared::detail::mapped_sha256(std::as_bytes(std::span(catalog.data(),catalog.size())))!=fields[8]||
      catalog.back()!='\n'||catalog.find('\r')!=catalog.npos)
    throw std::runtime_error("original GLM catalogue actual SHA/LF differs");
  const auto lines=split(catalog,'\n');
  if(lines.size()!=8||lines.front()!="HYGLME01\t1\t6"||!lines.back().empty())
    throw std::runtime_error("original GLM catalogue must contain exactly six authenticated original planes");
  std::array<source::FilePin,6> files;
  for(std::size_t i=0;i<files.size();++i) {
    const auto row=split(lines[i+1],'\t');const auto leg=static_cast<source::Leg>(i/2);const bool scales=i%2!=0;
    const auto description=source::source_description(leg,scales,model,contract.storage);
    const auto shape=description.dimensions();const auto bytes=source::shared::take(description.required_bytes(),"actual GLM expert plane extent");
    const std::string_view dtype=contract.storage==source::WeightStorage::kMxfp4Group32?"U8":(scales?"F32":"F8_E4M3");
    if(row.size()!=7||row[0]!=source::source_name(contract.identity,leg,scales,global)||!pin(row[2])||row[3]!=dtype||
        decimal(row[4])!=std::size_t(shape[0])||decimal(row[5])!=std::size_t(shape[1])||decimal(row[6])!=bytes)
      throw std::runtime_error("original GLM catalogue name/type/shape/bytes differs");
    files[i]={row[0],row[1],row[2],description};
  }
  return source::ExpertSet::load(directory,contract.identity,global,files,model,contract.storage);
}
struct Frontier final {
  Contract contract;
  std::filesystem::path root,request;
  bool wait=false;
  void ensure(std::shared_ptr<source::Data> data,std::span<const std::int32_t> globals) const {
    source::require_identity(contract.identity);
    if(!data||!pin(contract.plan_sha256)||data->identity()!=contract.identity||data->weight_storage()!=contract.storage)
      throw std::runtime_error("original GLM frontier/source artifact or storage differs");
    const auto started=std::chrono::steady_clock::now();bool announced=false;
    for(;;) {
      for(const auto id:data->missing(globals)) {
        auto decimal_id=std::to_string(id);if(decimal_id.size()<3)decimal_id.insert(0,3-decimal_id.size(),'0');
        const auto set=load_ready(root/("expert"+decimal_id),contract,id,data->model());if(set)data->attach_experts(set);
      }
      const auto missing=data->missing(globals);if(missing.empty())return;
      if(!announced) {
        if(!request.empty()) {
          if(!request.parent_path().empty())std::filesystem::create_directories(request.parent_path());
          auto temporary=request;temporary+=".tmp";std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
          const auto list=[&](auto ids){out<<'[';for(std::size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<']';};
          // Fields are fixed source-contract tokens, not arbitrary JSON text.
          for(const auto &value:{contract.identity.repository,contract.identity.revision,contract.identity.layer_prefix,contract.plan_sha256,contract.section})
            if(value.find_first_of("\"\\\r\n\t")!=value.npos)throw std::runtime_error("original GLM frontier identity has invalid characters");
          out<<"{\"section\":\""<<contract.section<<"\",\"repository\":\""<<contract.identity.repository<<"\",\"revision\":\""<<contract.identity.revision
             <<"\",\"layer_prefix\":\""<<contract.identity.layer_prefix<<"\",\"stage\":0,\"source_plan_sha256\":\""<<contract.plan_sha256
             <<"\",\"global_ids\":";list(globals);out<<",\"missing_ids\":";list(missing);out<<"}\n";out.close();
          if(!out)throw std::runtime_error("cannot finish actual GLM expert frontier");
#if defined(_WIN32)
          bool replaced=false;for(unsigned retry=0;retry<100;++retry) {
            if(MoveFileExW(temporary.c_str(),request.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){replaced=true;break;}
            const auto error=GetLastError();if(error!=ERROR_SHARING_VIOLATION&&error!=ERROR_ACCESS_DENIED)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          if(!replaced)throw std::runtime_error("cannot atomically replace actual GLM expert frontier");
#else
          std::filesystem::rename(temporary,request);
#endif
        }
        std::printf("actual GLM %s selected missing=%zu (%s)\n",contract.identity.layer_prefix.c_str(),missing.size(),wait?"explicit supply wait":"cache-only failure");
        std::fflush(stdout);announced=true;
      }
      if(!wait)throw std::runtime_error("actual original GLM experts missing; explicit supply required");
      if(std::chrono::steady_clock::now()-started>std::chrono::minutes(20))throw std::runtime_error("explicit original GLM expert wait exceeded20minutes");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
} // namespace glm_original_expert_frontier
