#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <array>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "shader_module.hpp"

constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool vk_enable_validation_layers = false;
#else
constexpr bool vk_enable_validation_layers = true;
#endif

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

[[nodiscard]] auto is_physical_device_suitable(const vk::PhysicalDevice &
                                               /*device*/) -> bool
{
  // TODO: Implement this
  return true;
}

auto pick_physical_device(const vk::Instance& instance)
    -> std::optional<vk::PhysicalDevice>
{
  const auto devices = instance.enumeratePhysicalDevices();
  assert(!devices.empty());

  for (const auto& device : devices) {
    if (is_physical_device_suitable(device)) {
      return device;
    }
  }

  return std::nullopt;
}

static VKAPI_ATTR auto VKAPI_CALL vk_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void *
    /*pUserData*/) -> VkBool32
{
  fmt::print(stdout, "validation layer: {}\n", pCallbackData->pMessage);
  std::fflush(stdout);
  return VK_FALSE;
}

auto check_validation_layer_support() -> bool
{
  const auto avaiable_layers = vk::enumerateInstanceLayerProperties();
  bool result = true;

  for (const char* layerName : validation_layers) {
    bool layer_found = false;

    for (const auto& layer_properties : avaiable_layers) {
      if (strcmp(layerName,
                 static_cast<const char*>(layer_properties.layerName)) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      fmt::print(stderr, "Require Validation layer ({}) not found\n",
                 layerName);
      result = false;
    }
  }

  return result;
}

[[nodiscard]] auto get_required_extensions() -> std::vector<const char*>
{
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions(glfw_extensions,
                                      glfw_extensions + glfw_extension_count);

  if (vk_enable_validation_layers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

[[nodiscard]] auto create_instance() -> vk::UniqueInstance
{
  if (vk_enable_validation_layers) {
    check_validation_layer_support();
  }

  vk::ApplicationInfo app_info;
  app_info.setApiVersion(VK_API_VERSION_1_1);

  const auto extensions = get_required_extensions();

  vk::InstanceCreateInfo create_info;
  create_info.setPApplicationInfo(&app_info)
      .setEnabledExtensionCount(static_cast<std::uint32_t>(extensions.size()))
      .setPpEnabledExtensionNames(extensions.data());
  if (vk_enable_validation_layers) {
    create_info.setEnabledLayerCount(validation_layers.size())
        .setPpEnabledLayerNames(validation_layers.data());
  } else {
    create_info.setEnabledLayerCount(0);
  }

  return vk::createInstanceUnique(create_info);
}

[[nodiscard]] auto create_dynamic_loader(const vk::Instance& instance)
    -> vk::DispatchLoaderDynamic
{
  vk::DispatchLoaderDynamic dldy;
  dldy.init(instance);
  return dldy;
}

auto setup_debug_messenger(const vk::Instance& instance,
                           const vk::DispatchLoaderDynamic& dldy)
    -> vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>
{
  vk::DebugUtilsMessengerCreateInfoEXT create_info;
  create_info
      .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                          vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                          vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose)
      .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
      .setPfnUserCallback(vk_debug_callback);

  return instance.createDebugUtilsMessengerEXTUnique(create_info, nullptr,
                                                     dldy);
}

auto get_compute_queue_index(const vk::PhysicalDevice& pd)
    -> std::optional<std::uint32_t>
{
  const auto properties = pd.getQueueFamilyProperties();
  for (std::uint32_t i = 0; i < properties.size(); ++i) {

    if (properties[i].queueFlags & vk::QueueFlagBits::eCompute) {
      return i;
    }
  }

  return std::nullopt;
}

auto vk_malloc(const vk::PhysicalDevice& pd, const vk::Device& device,
               VkDeviceSize memory_size)
{
  uint32_t memory_type_index = VK_MAX_MEMORY_TYPES;
  const auto properties = pd.getMemoryProperties();
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const auto memory_type = properties.memoryTypes[i];

    if ((memory_type.propertyFlags &
         vk::MemoryPropertyFlagBits::eHostVisible) &&
        (memory_type.propertyFlags &
         vk::MemoryPropertyFlagBits::eHostCoherent) &&
        memory_size < properties.memoryHeaps[memory_type.heapIndex].size) {
      memory_type_index = i;
      break;
    }
  }
  if (memory_type_index == VK_MAX_MEMORY_TYPES) {
    throw std::runtime_error("GPU out of memory");
  }

  const vk::MemoryAllocateInfo malloc_info{memory_size, memory_type_index};
  return device.allocateMemoryUnique(malloc_info);
}

int main() try {
  const auto instance = create_instance();
  [[maybe_unused]] const auto dldy = create_dynamic_loader(*instance);

#ifndef NDEBUG
  const auto debug_messenger = setup_debug_messenger(*instance, dldy);
#endif

  const auto pd = pick_physical_device(*instance).value();
  const auto compute_queue_family_index = get_compute_queue_index(pd).value();

  float queue_priorities = 1;
  const vk::DeviceQueueCreateInfo queue_create_info{
      {}, compute_queue_family_index, 1, &queue_priorities};
  const vk::DeviceCreateInfo create_info{{}, 1, &queue_create_info};

  const auto device = pd.createDeviceUnique(create_info);
  const auto compute_queue = device->getQueue(compute_queue_family_index, 0);

  // Create buffers
  constexpr uint32_t buffer_size = sizeof(int32_t) * (2 << 13);
  constexpr VkDeviceSize memory_size = buffer_size * 2;

  auto memory = vk_malloc(pd, *device, memory_size);

  std::random_device rd;
  auto* payload = static_cast<int*>(device->mapMemory(*memory, 0, memory_size));
  for (uint32_t k = 0; k < memory_size / sizeof(int32_t); ++k) {
    std::uniform_int_distribution<int> dis;
    payload[k] = dis(rd);
  }
  device->unmapMemory(*memory);

  vk::BufferCreateInfo buffer_create_info{
      {},
      buffer_size,
      vk::BufferUsageFlagBits::eStorageBuffer,
      vk::SharingMode::eExclusive,
      1,
      &compute_queue_family_index};

  const auto in_buffer = device->createBufferUnique(buffer_create_info);
  const auto in_memory_requirement =
      device->getBufferMemoryRequirements(*in_buffer);
  device->bindBufferMemory(*in_buffer, *memory, 0);

  const auto out_buffer = device->createBufferUnique(buffer_create_info);
  const auto out_memory_requirement =
      device->getBufferMemoryRequirements(*out_buffer);
  device->bindBufferMemory(*out_buffer, *memory, buffer_size);

  // Create Shader
  const auto shader_module =
      create_shader_module("shaders/copy.comp.spv", *device);

  // Creates descriptor set layout
  const vk::DescriptorSetLayoutBinding descriptor_set_layout_bindings[2] = {
      {0, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute, nullptr},
      {1, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute, nullptr}};

  const vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
      {}, 2, descriptor_set_layout_bindings};

  const auto descriptor_set_layout = device->createDescriptorSetLayoutUnique(
      descriptor_set_layout_create_info);

  // Create compute pipeline
  const vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
      {}, 1, &descriptor_set_layout.get()};
  const auto pipeline_layout =
      device->createPipelineLayoutUnique(pipeline_layout_create_info);

  vk::PipelineShaderStageCreateInfo shader_stage_create_info{
      {}, vk::ShaderStageFlagBits::eCompute, *shader_module, "main"};

  vk::ComputePipelineCreateInfo pipeline_create_info{
      {}, shader_stage_create_info, *pipeline_layout, {}, 0};

  const auto pipeline =
      device->createComputePipelineUnique({}, pipeline_create_info);

  // Create descriptor pool
  const vk::DescriptorPoolSize descriptor_pool_size{
      vk::DescriptorType::eStorageBuffer, 2};
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info{
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1,
      &descriptor_pool_size};
  const auto descriptor_pool =
      device->createDescriptorPoolUnique(descriptor_pool_create_info);

  // Allocates descriptor set
  const vk::DescriptorSetAllocateInfo descriptor_set_allocate_info{
      *descriptor_pool, 1, &descriptor_set_layout.get()};

  const auto descriptor_set =
      device->allocateDescriptorSetsUnique(descriptor_set_allocate_info);

  // Buffers
  const vk::DescriptorBufferInfo in_descriptor_buffer_info{*in_buffer, 0,
                                                           VK_WHOLE_SIZE};

  const vk::DescriptorBufferInfo out_descriptor_buffer_info{*out_buffer, 0,
                                                            VK_WHOLE_SIZE};

  // Command Pool
  const vk::CommandPoolCreateInfo command_pool_create_info{{}, 0};
  const auto command_pool =
      device->createCommandPoolUnique(command_pool_create_info);

  // Test Success
#if 0
  payload = static_cast<int*>(device->mapMemory(*memory, 0, memory_size));
  for (uint32_t k = 0, e = buffer_size / sizeof(int32_t); k < e; k++) {
    assert(payload[k + e] == payload[k]);
  }
  device->unmapMemory(*memory);
#endif

} catch (const std::exception& e) {
  fmt::print(stderr, "Error: {}\n", e.what());
} catch (...) {
  std::fputs("Unknown exception thrown!\n", stderr);
}
