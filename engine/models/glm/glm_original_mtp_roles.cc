#include "glm_original_mtp_roles.h"

namespace glm_original_mtp_roles {
void check(hyle::Status status,std::string_view where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
}

namespace glm_original_mtp_roles {
c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions,
                           c::DataType type) {
  return take(c::TensorDesc::dense(type,{dimensions.begin(),dimensions.size()}),"typed original description");
}
}

namespace glm_original_mtp_roles {
c::TensorDesc description(Parameter parameter) {
  switch(parameter) {
  case Parameter::kEmbeddingNorm:
  case Parameter::kHiddenNorm:
  case Parameter::kInputNorm:return dense({kHidden});
  case Parameter::kEhProjection:return dense({kHidden,kJoined});
  case Parameter::kQueryProjection:return take(c::float8_weight_description(kHidden,kQuery,
      c::Float8ScaleMode::kFloat32Block128),"original q_a packed type");
  case Parameter::kQueryNorm:return dense({kQuery});
  default:throw std::runtime_error("unknown original GLM prefix parameter");
  }
}
}

namespace glm_original_mtp_roles {
std::span<const std::byte> payload(const Data &data,Parameter parameter) {
  switch(parameter) {
  case Parameter::kEmbeddingNorm:return std::as_bytes(std::span(data.enorm));
  case Parameter::kHiddenNorm:return std::as_bytes(std::span(data.hnorm));
  case Parameter::kEhProjection:return std::as_bytes(std::span(data.eh));
  case Parameter::kInputNorm:return std::as_bytes(std::span(data.input_norm));
  case Parameter::kQueryProjection:return data.canonical;
  case Parameter::kQueryNorm:return std::as_bytes(std::span(data.query_norm));
  default:throw std::runtime_error("unknown original GLM prefix parameter");
  }
}
}

namespace glm_original_mtp_roles {
Prefix append(sem::Fragment &f,Value hidden,Value embedding,const Data &data) {
  return append_with_constants(f,hidden,embedding,[&](Parameter parameter) {
    const auto raw=payload(data,parameter);
    return f.constant(description(parameter),Bytes(raw.begin(),raw.end()));
  });
}
}
