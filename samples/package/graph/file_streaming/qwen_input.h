#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <hyle/compute/tensor_desc.h>
#include <hyle/error/result.h>

// Consumer of the official-tool checked manifest, not a Safetensors parser.
// Preparation binds original file/header/range identities and references.
namespace qwen_input {
namespace c = hyle::compute;
struct Range final {
  std::string name, digest;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
  c::TensorDesc description;
  std::vector<float> reference;
};
struct Manifest final {
  std::string file;
  std::uint64_t capacity = 0;
  std::array<Range, 4> ranges;
};
inline hyle::Result<Manifest> read(const char *path) {
  std::ifstream input(path);
  if (!input) return std::unexpected(hyle::Status::NotFound("checked Qwen range manifest"));
  std::string version; std::getline(input, version);
  if (version != "HYLE_F3_QWEN_RANGES_V1") return std::unexpected(hyle::Status::InvalidArgument("checked range version"));
  Manifest result; std::getline(input, result.file); input >> result.capacity;
  for (auto &range : result.ranges) {
    std::string dtype; std::int64_t rows = 0, columns = 0;
    if (!(input >> range.name >> dtype >> range.offset >> range.bytes >> rows >> columns >> range.digest))
      return std::unexpected(hyle::Status::InvalidArgument("incomplete checked tensor range"));
    if (dtype == "BF16") {
      const std::array dims{rows, columns}; HYLE_ASSIGN_OR_RETURN(range.description, c::TensorDesc::dense(c::DataType::kBFloat16, dims));
    } else if (dtype == "F32" && rows == 128 && columns == 1) {
      const std::array dims{rows}; HYLE_ASSIGN_OR_RETURN(range.description, c::TensorDesc::dense(c::DataType::kFloat32, dims));
    } else return std::unexpected(hyle::Status::InvalidArgument("unexpected checked mixed dtype"));
    HYLE_ASSIGN_OR_RETURN(auto required, range.description.required_bytes());
    if (required != range.bytes || range.offset > result.capacity || range.bytes > result.capacity - range.offset)
      return std::unexpected(hyle::Status::OutOfRange("checked Qwen tensor shape/range"));
    auto reference_name = range.name;
    for (auto &ch : reference_name) if (ch == '.') ch = '_';
    auto reference_path = std::filesystem::path(path).parent_path() / (reference_name + ".reference.txt");
    std::ifstream reference(reference_path); float value = 0;
    while (reference >> value) range.reference.push_back(value);
    if (range.reference.size() != static_cast<std::size_t>(rows))
      return std::unexpected(hyle::Status::InvalidArgument("checked Qwen reference size"));
  }
  return result;
}
} // namespace qwen_input
