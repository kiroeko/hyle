#include "glm_hyper_connection_fragment.h"

namespace glm_hyper_connection_fragment {
Value cast(sem::Fragment &f,Value input,c::DataType type) {
  return f.one(c::Operation(c::Cast{.output_type=type}),{input});
}
}

namespace glm_hyper_connection_fragment {
Value slice(sem::Fragment &f,Value input,std::int32_t axis,std::int64_t begin,std::int64_t count) {
  return f.one(c::Operation(c::View::slice(axis,begin,count).value()),{input});
}
}

namespace glm_hyper_connection_fragment {
Value compact(sem::Fragment &f,Value input) {return f.one(c::Operation(c::Copy{}),{input});}
}

namespace glm_hyper_connection_fragment {
Value reduce(sem::Fragment &f,Value input,std::int32_t axis,c::ReductionKind kind,bool keep) {
  return f.one(c::Operation(c::Reduce{.kind=kind,.axes=c::Axes::from(std::array{axis}).value(),
      .keep_dimensions=keep,.accumulation_type=c::DataType::kFloat32}),{input});
}
}

namespace glm_hyper_connection_fragment {
void require_storage(c::DataType type) {
  if(type!=c::DataType::kBFloat16&&type!=c::DataType::kFloat32)
    throw std::runtime_error("GLM HC requires its explicit BF16 or F32 activation storage");
}
}

namespace glm_hyper_connection_fragment {
Value as_f32(sem::Fragment &f,Value input) {
  const auto type=f.description(input).data_type();require_storage(type);
  return type==c::DataType::kFloat32?input:cast(f,input,c::DataType::kFloat32);
}
}

namespace glm_hyper_connection_fragment {
Value store(sem::Fragment &f,Value input,c::DataType type) {
  require_storage(type);
  if(f.description(input).data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("GLM explicit activation store needs its real F32 predecessor");
  return type==c::DataType::kFloat32?input:cast(f,input,type);
}
}

namespace glm_hyper_connection_fragment {
NormPorts norm(sem::Fragment &f,Value actual,Value gamma,
                      c::DataType storage_type) {
  require_storage(storage_type);
  if(f.description(actual).data_type()!=storage_type)
    throw std::runtime_error("GLM learned RMS input differs from its explicit activation storage");
  NormPorts result;result.input=as_f32(f,actual);
  result.raw=f.one(c::Operation(c::RmsNorm{.epsilon=kNormEpsilon}),{result.input,as_f32(f,gamma)});
  result.stored=store(f,result.raw,storage_type);
  result.visible=as_f32(f,result.stored);return result;
}
}

namespace glm_hyper_connection_fragment {
HcPorts hc_pre(sem::Fragment &f,Value residual,Value fn,Value scale,Value base,Value norm_weight,
                      c::DataType storage_type,
                      ProjectionOrder order) {
  const auto project=[](sem::Fragment &graph,Value actual_input,Value mixing) {
  if(graph.description(mixing).data_type()==c::DataType::kBFloat16) {
    return graph.one(c::model_ops::dense_linear_f32(),{actual_input,mixing});
  } else {
    if(graph.description(mixing).data_type()!=c::DataType::kFloat32)
      throw std::runtime_error("GLM HC mixing coefficients must retain actual BF16 or F32 storage");
    const auto transpose=graph.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{mixing});
    return graph.one(c::Operation(c::MatMul{}),{actual_input,transpose});
  }
  };
  return hc_pre_with_projection(f,residual,fn,scale,base,norm_weight,storage_type,order,project);
}
}

namespace glm_hyper_connection_fragment {
PostPorts hc_post(sem::Fragment &f,const HcPorts &pre,Value residual,Value branch,
                        c::DataType storage_type) {
  require_storage(storage_type);
  if(f.description(residual).data_type()!=storage_type||f.description(branch).data_type()!=storage_type)
    throw std::runtime_error("GLM HC post operands differ from explicit activation storage");
  PostPorts result;result.branch=branch;result.residual=residual;
  result.raw=f.one(c::model_ops::hyper_connection_post(),
      {as_f32(f,branch),as_f32(f,residual),pre.post,pre.combination});
  result.stored=store(f,result.raw,storage_type);
  result.visible=as_f32(f,result.stored);return result;
}
}
