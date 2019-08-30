
macro(add_compilation_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} linalg)
endmacro()

#add_compilation_test(ctest_thingy)

