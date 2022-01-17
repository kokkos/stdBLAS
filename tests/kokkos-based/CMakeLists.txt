
macro(linalg_add_test_kokkos TESTNAME FINDSTR)
  set(testNameFinal utest_${TESTNAME})
  set(testExe ${testNameFinal}_exe)
  add_executable(${testExe} ${TESTNAME}.cpp gtest_main_kokkos.cpp)
  target_link_libraries(${testExe} linalg GTest::GTest)

  add_test(
    NAME ${testNameFinal}
    COMMAND ${CMAKE_COMMAND}
    -DEXE_NAME=$<TARGET_FILE:${testExe}>
    -DALGO_NAME=${TESTNAME}
    -DLOG_FILE=${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.log
    -DTEST_STRING_FIND=${FINDSTR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake)

    # we have to set fail regex so that if it fails,
    # test.cmake will detect it
    set_tests_properties(
      ${testNameFinal} PROPERTIES
      FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;FAILED")
endmacro()

linalg_add_test_kokkos(
  add_kokkos
  "add: kokkos impl")
linalg_add_test_kokkos(
  scale_kokkos
  "scale: kokkos impl")
linalg_add_test_kokkos(
  dot_kokkos
  "dot: kokkos impl")
linalg_add_test_kokkos(
  dotc_kokkos
  "dot: kokkos impl")
linalg_add_test_kokkos(
  vector_abs_sum_kokkos
  "vector_abs_sum: kokkos impl")
linalg_add_test_kokkos(
  idx_abs_max_kokkos
  "idx_abs_max: kokkos impl")
linalg_add_test_kokkos(
  vector_norm2_kokkos
  "vector_norm2: kokkos impl")
linalg_add_test_kokkos(
  vector_sum_of_squares_kokkos
  "vector_sum_of_squares: kokkos impl")
linalg_add_test_kokkos(
  copy_kokkos
  "copy: kokkos impl")
linalg_add_test_kokkos(
  swap_elements_kokkos
  "swap_elements: kokkos impl")
linalg_add_test_kokkos(
  matrix_frob_norm_kokkos
  "matrix_frob_norm: kokkos impl")
