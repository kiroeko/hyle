#include "dspark_original_expert_roles.h"

namespace dspark_original_expert_roles {
Selection select(std::span<const std::int32_t> actual_ids) {
  if(actual_ids.empty()||actual_ids.size()>30)throw std::runtime_error("actual expert selection extent");
  Selection result;result.map.fill(-1);
  for(const auto id:actual_ids) {
    if(id<0||id>=256)throw std::runtime_error("actual global expert ID outside original domain");
    result.globals.push_back(id);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
  return result;
}
}

namespace dspark_original_expert_roles {
std::size_t index(std::size_t stage,Leg leg) {
  const auto value=std::size_t(leg);
  if(stage>=3||value>=3)throw std::runtime_error("original expert role outside stage/domain");
  return stage*3+value;
}
}

namespace dspark_original_expert_roles {
c::TensorDesc resident_description(Leg leg,std::int64_t capacity) {
  if(std::size_t(leg)>=3||capacity<=0||capacity>256)throw std::runtime_error("resident expert geometry");
  const bool down=leg==Leg::kDown;
  auto result=c::gguf_grouped_weight_description(c::GgufQuantizedType::kMxfp4,c::DataType::kFloat32,
      capacity,down?4096:2048,down?2048:4096);
  if(!result)throw std::runtime_error(std::string(result.error().message()));return *result;
}
}

namespace dspark_original_expert_roles {
std::string mapped_sha256(std::span<const std::byte> input) {
  constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
  std::array<std::uint32_t, 8> hash{
      0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
      0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  if(input.size()>std::numeric_limits<std::uint64_t>::max()/8)throw std::runtime_error("mapped SHA length overflow");
  const auto consume=[&](std::span<const std::byte,64> block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i)
      for (std::size_t byte = 0; byte < 4; ++byte)
        words[i] = (words[i] << 8u) |
            std::to_integer<std::uint32_t>(block[i * 4 + byte]);
    for (std::size_t i = 16; i < words.size(); ++i) {
      const auto x = words[i - 15], y = words[i - 2];
      const auto small0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3u);
      const auto small1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10u);
      words[i] = words[i - 16] + small0 + words[i - 7] + small1;
    }
    auto a=hash[0], b=hash[1], cc=hash[2], d=hash[3];
    auto e=hash[4], f=hash[5], gg=hash[6], h=hash[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto big1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto first = h + big1 + ((e & f) ^ (~e & gg)) + constants[i] + words[i];
      const auto big0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto second = big0 + ((a & b) ^ (a & cc) ^ (b & cc));
      h=gg; gg=f; f=e; e=d+first; d=cc; cc=b; b=a; a=first+second;
    }
    hash[0]+=a; hash[1]+=b; hash[2]+=cc; hash[3]+=d;
    hash[4]+=e; hash[5]+=f; hash[6]+=gg; hash[7]+=h;
  };
  std::size_t offset=0;
  for(;input.size()-offset>=64;offset+=64)consume(std::span<const std::byte,64>(input.data()+offset,64));
  std::array<std::byte,128> tail{};const auto remaining=input.size()-offset;
  std::copy(input.begin()+std::ptrdiff_t(offset),input.end(),tail.begin());tail[remaining]=std::byte{0x80};
  const auto extent=remaining<56?64u:128u;const auto bits=std::uint64_t(input.size())*8;
  for(std::size_t byte=0;byte<8;++byte)tail[extent-1-byte]=std::byte((bits>>(byte*8))&0xffu);
  consume(std::span<const std::byte,64>(tail.data(),64));
  if(extent==128)consume(std::span<const std::byte,64>(tail.data()+64,64));
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto word : hash) result << std::setw(8) << word;
  return result.str();
}
}

namespace dspark_original_expert_roles {
Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source)throw std::runtime_error("original expert source missing");
  Roles result{.source=std::move(source)};
  const auto immutable=[&](std::string_view name,const c::TensorDesc &description,std::span<const std::byte> raw) {
    auto backing=m::Buffer::create(owner.ref(),raw.size(),256);
    if(!backing)throw std::runtime_error(std::string(backing.error().message()));
    auto status=backing->view().write(raw);if(!status.ok())throw std::runtime_error(std::string(status.message()));
    auto lease=owner.adopt_buffer(std::move(*backing));if(!lease)throw std::runtime_error(std::string(lease.error().message()));
    return b::ReadonlyRole{std::string(name),description,owner,std::move(*lease)};
  };
  const auto router=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{256,4096}).value();
  const auto bias=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{256}).value();
  for(std::size_t stage=0;stage<3;++stage) {
    result.routers[stage]=immutable(kRouters[stage].name,router,std::as_bytes(std::span(result.source->routers[stage])));
    result.biases[stage]=immutable(kBiases[stage].name,bias,std::as_bytes(std::span(result.source->biases[stage])));
  }
  return result;
}
}

namespace dspark_original_expert_roles {
hyle::Status qualify(const Roles &roles,const b::TargetIdentity &target) {
  if(!roles.source||target.model_repository!=b::kModelRepository||target.model_revision!=b::kModelRevision)
    return hyle::Status::InvalidArgument("original routed experts require formal Flash0731 identity");
  const auto router=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{256,4096}).value();
  const auto bias=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{256}).value();
  for(std::size_t stage=0;stage<3;++stage) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.routers[stage],kRouters[stage].name,router));
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.biases[stage],kBiases[stage].name,bias));
  }
  return hyle::Status::Ok();
}
}
