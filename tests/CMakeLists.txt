
macro(linalg_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} linalg GTest::GTest GTest::Main ${BLAS_LIBRARIES})
  add_test(${name} ${name})
endmacro()

find_package(GTest REQUIRED)

linalg_add_test(add)
linalg_add_test(conjugate_transpose_view)
linalg_add_test(conjugate_view)
linalg_add_test(copy)
linalg_add_test(dot)
linalg_add_test(givens)
linalg_add_test(norm2)
linalg_add_test(scaled_view)
linalg_add_test(swap)
linalg_add_test(transpose_view)
