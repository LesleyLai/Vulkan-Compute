#include <fstream>
#include <string>

#include "shader_module.hpp"

[[nodiscard]] static auto read_file(const std::string& filename)
    -> std::vector<char>
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filename);
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer;
  buffer.resize(file_size);

  file.seekg(0);
  file.read(buffer.data(), file_size);

  return buffer;
}

[[nodiscard]] auto create_shader_module(const std::string& filename,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule
{
  return create_shader_module(read_file(filename), device);
}

[[nodiscard]] auto create_shader_module(const std::vector<char>& code,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule
{
  vk::ShaderModuleCreateInfo create_info{
      {}, code.size(), reinterpret_cast<const std::uint32_t*>(code.data())};

  return device.createShaderModuleUnique(create_info);
}
