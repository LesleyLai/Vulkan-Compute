#ifndef SHADER_MODULE_HPP
#define SHADER_MODULE_HPP

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

[[nodiscard]] auto create_shader_module(const std::string& filename,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule;

[[nodiscard]] auto create_shader_module(const std::vector<char>& code,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule;

#endif // SHADER_MODULE_HPP
