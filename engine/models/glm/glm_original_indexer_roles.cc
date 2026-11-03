#include "glm_original_indexer_roles.h"

namespace glm_original_indexer_roles {
std::span<const std::uint16_t> values(const Data &data, Parameter parameter) {
  switch (parameter) {
  case kApe: return data.ape;
  case kGate: return data.gate;
  case kBias: return data.bias;
  case kNorm: return data.norm;
  case kWeights: return data.weights;
  case kKey: return data.key;
  case kQuery: return data.query;
  default: throw std::runtime_error("unknown original GLM indexer role");
  }
}
}

namespace glm_original_indexer_roles {
c::TensorDesc description(Parameter parameter) {
  switch (parameter) {
  case kApe: return sem::dense({1, 4, 128});
  case kGate: case kKey: return sem::dense({128, 4096}, c::DataType::kBFloat16);
  case kBias: case kNorm: return sem::dense({128});
  case kWeights: return sem::dense({32, 4096});
  case kQuery: return sem::dense({4096, 1536}, c::DataType::kBFloat16);
  default: throw std::runtime_error("unknown original GLM indexer description");
  }
}
}

namespace glm_original_indexer_roles {
std::vector<float> widen(std::span<const std::uint16_t> source) {
  std::vector<float> result(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) result[i] = original::widen(source[i]);
  return result;
}
}

namespace glm_original_indexer_roles {
Bytes payload(const Data &data, Parameter parameter) {
  const auto source = values(data, parameter);
  if (description(parameter).data_type() == c::DataType::kBFloat16) return sem::bytes<std::uint16_t>(source);
  return sem::bytes(widen(source));
}
}

namespace glm_original_indexer_roles {
Roles bind(const Data &data, const c::StorageOwner &owner) {
  Roles result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto parameter = static_cast<Parameter>(i);
    const auto content = payload(data, parameter);
    auto buffer = take(m::Buffer::create(owner.ref(), content.size(), 256), "original indexer immutable bank");
    const auto status = buffer.view().write(content);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    result[i] = {kNames[i], description(parameter), owner,
        take(owner.adopt_buffer(std::move(buffer)), "original indexer readonly owner")};
  }
  return result;
}
}
