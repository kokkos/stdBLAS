
macro(add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} linalg GTest::GTest GTest::Main)
endmacro()

find_package(GTest REQUIRED)

add_test(conjugate_view)
add_test(dot)

