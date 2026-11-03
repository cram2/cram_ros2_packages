cmake_minimum_required(VERSION 3.8)
project(robokudo_cad_data)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
