#pragma once
// Fixed actual C5 packing/geometry demand; no execution PASS.
namespace cpu_c5_non_gguf_inputs {
enum class Format { kNf4, kAwq, kCt, kAffine, kMx4, kMx8 };
struct Artifact { const char *repository,*revision,*digest; };
struct Case { const char *id; Format format; std::int64_t e,n,k; int group; bool grouped,embedding; unsigned artifact; };
inline constexpr Artifact artifacts[]{
  {"techwithsergiu/Qwen3.5-2B-bnb-4bit","97fb2bcc0255f1d5b36f07daa54b76cb29444df6","29762312eb2316561c75fbbf8f9d1b58b2ebd4f82741fa3cfbfac9829829feaa"},
  {"mattbucci/Qwen3.8-27B-AWQ","2cc4cffb391645c9be52f3fe50ff595ac7df3e38","c50d83a710d3c9fab0f6aac8e27b105ea4a20c8ccf15e43f0b7559f95764dc79"},
  {"soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ","a1b8f1570ecc09758c8431fd536c88e3b209674f","0fb1ddf1bfef6f0acd5f6e5e5f4d63f70f1f412af8d593619e8471455fe4e8c5"},
  {"ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit","c8b8f3fe9112cc5aa3b34d2aec9f771b68622902","dd2dd8d64fce6a82c09a3561cf32656ddfa8090334d3524a76e922a4efaf095b"},
};
inline constexpr Case cases[]{
  {"n0bb663b75e961054",Format::kMx8,1,32768,1024,32,false,false,3},
  {"n11ac28dbeaa21a76",Format::kMx4,256,2048,4096,32,true,false,3},
  {"n1a02d912dccfbfcc",Format::kMx8,1,512,4096,32,false,false,3},
  {"n202aedd266175231",Format::kMx8,1,4096,8192,32,false,false,3},
  {"n2208fbb06e447168",Format::kAwq,1,1024,5120,128,false,false,1},
  {"n2af37225831ba519",Format::kNf4,1,16,2048,64,false,false,0},
  {"n33d5fe29fed560af",Format::kCt,1,1024,5120,128,false,false,2},
  {"n37e21568674f2571",Format::kMx8,1,2048,4096,32,false,false,3},
  {"n39f10794d1034fdd",Format::kCt,1,17408,5120,128,false,false,2},
  {"n44af3ac14c76c678",Format::kMx8,1,1024,4096,32,false,false,3},
  {"n57f8346172bcde91",Format::kAffine,1,256,4096,64,false,false,3},
  {"n5b450b9128f9ddb4",Format::kAffine,1,512,4096,64,false,false,3},
  {"n627e13f695902a6f",Format::kAwq,1,5120,17408,128,false,false,1},
  {"n7251e93056484f69",Format::kAffine,1,1024,4096,64,false,false,3},
  {"n7b3b05721f557885",Format::kAwq,1,6144,5120,128,false,false,1},
  {"n7b87eddf86e971d2",Format::kNf4,1,2048,2048,64,false,false,0},
  {"n827da62ad0daf7b8",Format::kMx8,1,8192,1024,32,false,false,3},
  {"n85a9768ce0cfbe9d",Format::kAffine,1,64,4096,64,false,false,3},
  {"n8c79bf7f55b10c01",Format::kAwq,1,12288,5120,128,false,false,1},
  {"n9bfaa0e6dba16c70",Format::kMx8,1,4096,2048,32,false,false,3},
  {"na2bfd546f5de3726",Format::kAffine,1,129280,4096,64,false,true,3},
  {"na7b682381e9fa396",Format::kCt,1,5120,17408,128,false,false,2},
  {"nab5612104243530d",Format::kNf4,1,4096,2048,64,false,false,0},
  {"nad972a78d88299e9",Format::kCt,1,5120,6144,128,false,false,2},
  {"nb0eb5191fc5bb45c",Format::kNf4,1,512,2048,64,false,false,0},
  {"nb696beddeed9063f",Format::kAffine,1,129280,4096,64,false,false,3},
  {"nb79fa3dba1813475",Format::kCt,1,12288,5120,128,false,false,2},
  {"nc0c493d5f357c6f5",Format::kAwq,1,10240,5120,128,false,false,1},
  {"nc1901694d1ed5be9",Format::kNf4,1,2048,6144,64,false,false,0},
  {"nc801b7e28c4a66c0",Format::kMx4,256,4096,2048,32,true,false,3},
  {"ncb8979d217d13ad8",Format::kNf4,1,6144,2048,64,false,false,0},
  {"ne3db6edc9b6e92ec",Format::kMx8,8,1024,4096,32,true,false,3},
  {"nf425aabf7fb25c64",Format::kAwq,1,17408,5120,128,false,false,1},
  {"nfab7af12a04d1cd9",Format::kAwq,1,5120,6144,128,false,false,1},
};
} // namespace cpu_c5_non_gguf_inputs
