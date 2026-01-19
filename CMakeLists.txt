cmake_minimum_required(VERSION 3.10)
project(legged_model VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wno-return-type -Wno-deprecated-declarations)

##############################
# 1️⃣ ROS1 / Catkin 支持
##############################
find_package(catkin QUIET)
if(catkin_FOUND)
  message(STATUS "Found catkin")
    catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
    )
endif()

##############################
# 2️⃣ 纯 C++ 依赖
##############################

# 第三方库
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(manif REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(logger REQUIRED)

##############################
# 3️⃣ 创建库（导出给上游用）
##############################
add_library(${PROJECT_NAME} STATIC
    src/Rotation.cpp
    src/LeggedState.cpp
    src/LeggedModel.cpp
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${manif_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
    yaml-cpp
    logger::CsvLogger
)

##############################
# 4️⃣ 可执行文件（独立测试）
##############################
add_executable(Math_test test/Math_test.cpp)
target_link_libraries(Math_test PRIVATE ${PROJECT_NAME})

add_executable(Rotation_test test/Rotation_test.cpp)
target_link_libraries(Rotation_test PRIVATE ${PROJECT_NAME})

add_executable(Lie_test test/Lie_test.cpp)
target_link_libraries(Lie_test PRIVATE ${PROJECT_NAME})

add_executable(LeggedState_test test/LeggedState_test.cpp)
target_link_libraries(LeggedState_test PRIVATE ${PROJECT_NAME})

add_executable(LeggedModel_test test/LeggedModel_test.cpp)
target_link_libraries(LeggedModel_test PRIVATE ${PROJECT_NAME})

##############################
# 5️⃣ 安装：库 + 头文件 + CMake Package（关键）
##############################
# 安装库
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 导出 targets 文件（CMake 自动生成）
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# 生成并安装 Config.cmake + ConfigVersion.cmake（你需要提供 .in 模板文件）
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

##############################
# 5️⃣ ROS1 测试与安装
##############################
if(catkin_FOUND)
    # 安装库和头文件
    install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    )
endif()

##############################
# 6️⃣ ROS2 / ament_cmake 支持
##############################
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
    message(STATUS "Found ament_cmake")

    # 安装可执行文件
    install(TARGETS Math_test Rotation_test Lie_test LeggedState_test LeggedModel_test
        DESTINATION lib/${PROJECT_NAME}
    )

    # 这两行让 ament 生态下的下游也能拿到 targets/依赖（推荐）
    ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
    ament_export_dependencies(Eigen3 pinocchio manif yaml-cpp logger)

    ament_package()
endif()
