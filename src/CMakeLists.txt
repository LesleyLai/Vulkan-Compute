find_package(Vulkan)

add_executable(VulkanRenderer "main.cpp"
    "shader_module.hpp" "shader_module.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    Vulkan::Vulkan
    CONAN_PKG::fmt
    CONAN_PKG::glfw
    )

include(CompileShader)
compile_shader(shader
   SOURCE ${CMAKE_SOURCE_DIR}/shaders/copy.comp
   TARGET ${CMAKE_BINARY_DIR}/bin/shaders/copy.comp.spv
)

set_target_properties(VulkanRenderer PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin")
add_dependencies(VulkanRenderer shader)
