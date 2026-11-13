#pragma once

#include <charconv>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>
#include "prompt_attachment.h"

namespace hyle::inference {
// The token field may have ordered, opaque conditioning records after ';':
// ids;format:begin:rows:shape:base64[;...]. No filenames/native handles cross
// this local transport. Plain token fields retain their original wire bytes.
inline bool parse_prompt_attachments(std::string_view field, std::size_t prompt_rows,
                                    std::vector<PromptAttachment> &result) {
  const auto first = field.find(';');
  if (first == std::string_view::npos) return true;
  field.remove_prefix(first+1);
  std::size_t end = 0, total = 0;
  const auto number = [](std::string_view word, std::size_t &value) {
    const auto parsed=std::from_chars(word.data(),word.data()+word.size(),value);
    return parsed.ec==std::errc{} && parsed.ptr==word.data()+word.size();
  };
  while (!field.empty()) {
    const auto next=field.find(';'); auto record=field.substr(0,next);
    std::array<std::string_view,5> parts;
    for(std::size_t i=0;i<4;++i){
      const auto colon=record.find(':');if(colon==std::string_view::npos)return false;
      parts[i]=record.substr(0,colon);record.remove_prefix(colon+1);
    }
    parts[4]=record;
    PromptAttachment attachment;
    if(parts[0].empty() || parts[0].size()>128)return false;
    for(const unsigned char ch:parts[0])
      if(!((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')||ch=='.'||ch=='_'||ch=='-'))return false;
    attachment.format=parts[0];
    if(!number(parts[1],attachment.begin)||!number(parts[2],attachment.rows)||
        attachment.begin<end||attachment.begin>prompt_rows||attachment.rows>prompt_rows-attachment.begin)return false;
    end=attachment.begin+attachment.rows;
    auto shape=parts[3];
    while(!shape.empty()){
      const auto comma=shape.find(',');std::size_t value=0;
      if(attachment.shape.size()==32||!number(shape.substr(0,comma),value)||!value||value>INT32_MAX)return false;
      attachment.shape.push_back(value);
      if(comma==std::string_view::npos)break;
      shape.remove_prefix(comma+1);if(shape.empty())return false;
    }
    if(attachment.shape.empty()||record.empty()||record.size()%4)return false;
    constexpr std::size_t maximum=128u<<20;
    const auto padding=record.ends_with("==")?2u:record.ends_with('=')?1u:0u;
    const auto bytes=record.size()/4*3-padding;
    if(!bytes||bytes>maximum-total)return false;
    attachment.payload.reserve(bytes);
    const auto decode=[](unsigned char ch)->int {
      if(ch>='A'&&ch<='Z')return ch-'A';if(ch>='a'&&ch<='z')return ch-'a'+26;
      if(ch>='0'&&ch<='9')return ch-'0'+52;if(ch=='+')return 62;if(ch=='/')return 63;return -1;
    };
    for(std::size_t at=0;at<record.size();at+=4){
      std::uint32_t value=0;
      for(std::size_t i=0;i<4;++i){
        const bool pad=at+i>=record.size()-padding;
        const auto digit=pad?0:decode(static_cast<unsigned char>(record[at+i]));
        if(digit<0||(pad&&record[at+i]!='='))return false;
        value=(value<<6)|static_cast<std::uint32_t>(digit);
      }
      if(at+4==record.size() && ((padding==1 && (value&0xffu)) || (padding==2 && (value&0xffffu))))return false;
      for(int shift=16;shift>=0;shift-=8)
        if(attachment.payload.size()<bytes)attachment.payload.push_back(static_cast<std::byte>((value>>shift)&255));
    }
    total+=bytes;result.push_back(std::move(attachment));
    if(next==std::string_view::npos)return true;
    field.remove_prefix(next+1);if(field.empty())return false;
  }
  return false;
}
} // namespace hyle::inference
