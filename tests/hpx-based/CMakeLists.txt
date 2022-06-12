#  Copyright (c) 2022 Hartmut Kaiser

macro(linalg_add_test_hpx TESTNAME FINDSTR)

  # the macro might have extra args
  set(appendToTestName "")
  set(compileDefs "")
  set(extra_args "${ARGN}")
  list(LENGTH extra_args extra_count)
  if (${extra_count} EQUAL 2)
    list(GET extra_args 0 compileDefs)
    list(GET extra_args 1 appendToTestName)
  endif()

  if (appendToTestName STREQUAL "")
    set(testNameFinal utest_${TESTNAME})
  else()
    set(testNameFinal utest_${TESTNAME}_${appendToTestName})
  endif()
  set(testExe ${testNameFinal}_exe)
  add_executable(${testExe} ${TESTNAME}.cpp gtest_main_hpx.cpp)
  target_link_libraries(${testExe} linalg GTest::GTest HPX::hpx)

  if (NOT compileDefs STREQUAL "")
    target_compile_definitions(${testExe} PRIVATE -D${compileDefs} -DHPX_STDBLAS_ENABLE_TESTS)
  else()
    target_compile_definitions(${testExe} PRIVATE -DHPX_STDBLAS_ENABLE_TESTS)
  endif()

  set_target_properties(${testExe} PROPERTIES FOLDER "Tests/HPX")

  add_test(
    NAME ${testNameFinal}
    COMMAND ${CMAKE_COMMAND}
    -DEXE_NAME=$<TARGET_FILE:${testExe}>
    -DALGO_NAME=${TESTNAME}
    -DLOG_FILE=${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}_${appendToTestName}.log
    -DTEST_STRING_FIND=${FINDSTR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake)

    # we have to set fail regex so that if it fails,
    # test.cmake will detect it
    set_tests_properties(
      ${testNameFinal} PROPERTIES
      FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;FAILED")
endmacro()

# add tests below

#
# blas1 (according to P1673)
#

#linalg_add_test_hpx(
#  dot_hpx
#  "dot: hpx impl")
#linalg_add_test_hpx(
#  dotc_hpx
#  "dot: hpx impl") # this his not a typo, dotc calls dot underneath

linalg_add_test_hpx(
  add_rank1_hpx
  "add: hpx impl")

linalg_add_test_hpx(
  scale_rank1_hpx
  "scale: hpx impl")
linalg_add_test_hpx(
  scale_rank2_hpx
  "scale: hpx impl")

#linalg_add_test_hpx(
#  idx_abs_max_hpx
#  "idx_abs_max: hpx impl")

#linalg_add_test_hpx(
#  vector_norm2_hpx
#  "vector_norm2: hpx impl")
#linalg_add_test_hpx(
#  vector_sum_of_squares_hpx
#  "vector_sum_of_squares: hpx impl")
#
#linalg_add_test_hpx(
#  vector_abs_sum_hpx
#  "vector_abs_sum: hpx impl")

#linalg_add_test_hpx(
#  matrix_frob_norm_hpx
#  "matrix_frob_norm: hpx impl")
#linalg_add_test_hpx(
#  matrix_one_norm_hpx
#  "matrix_one_norm: hpx impl")
#linalg_add_test_hpx(
#  matrix_inf_norm_hpx
#  "matrix_inf_norm: hpx impl")

#linalg_add_test_hpx(
#  swap_elements_rank1_hpx
#  "swap_elements: hpx impl")
#linalg_add_test_hpx(
#  swap_elements_rank2_hpx
#  "swap_elements: hpx impl")
#linalg_add_test_hpx(
#  copy_hpx
#  "copy: hpx impl")

#
# blas2 (according to P1673)
#

#linalg_add_test_hpx(
#  overwriting_matrix_vector_product
#  "overwriting_matrix_vector_product: hpx impl")
#linalg_add_test_hpx(
#  updating_matrix_vector_product
#  "updating_matrix_vector_product: hpx impl")

# symmetric_matrix_vector_product
#linalg_add_test_hpx(
#  overwriting_symmetric_matrix_vector_product
#  "overwriting_symmetric_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  overwriting_symmetric_matrix_vector_product
#  "overwriting_symmetric_matrix_vector_product_upper: hpx impl" USE_UPPER upper)
#linalg_add_test_hpx(
#  updating_symmetric_matrix_vector_product
#  "updating_symmetric_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  updating_symmetric_matrix_vector_product
#  "updating_symmetric_matrix_vector_product_upper: hpx impl" USE_UPPER upper)

# hermitian_matrix_vector_product
#linalg_add_test_hpx(
#  overwriting_hermitian_matrix_vector_product
#  "overwriting_hermitian_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  overwriting_hermitian_matrix_vector_product
#  "overwriting_hermitian_matrix_vector_product_upper: hpx impl" USE_UPPER upper)
#linalg_add_test_hpx(
#  updating_hermitian_matrix_vector_product
#  "updating_hermitian_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  updating_hermitian_matrix_vector_product
#  "updating_hermitian_matrix_vector_product_upper: hpx impl" USE_UPPER upper)

# triangular_matrix_vector_product
#linalg_add_test_hpx(
#  overwriting_triangular_matrix_vector_product
#  "overwriting_triangular_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  overwriting_triangular_matrix_vector_product
#  "overwriting_triangular_matrix_vector_product_upper: hpx impl" USE_UPPER upper)
#linalg_add_test_hpx(
#  updating_triangular_matrix_vector_product
#  "updating_triangular_matrix_vector_product_lower: hpx impl" USE_LOWER lower)
#linalg_add_test_hpx(
#  updating_triangular_matrix_vector_product
#  "updating_triangular_matrix_vector_product_upper: hpx impl" USE_UPPER upper)
#
#linalg_add_test_hpx(
#  matrix_rank1_update_hpx
#  "matrix_rank1_update: hpx impl")
#
#linalg_add_test_hpx(
#  symmetric_matrix_rank1_update_hpx
#  "symmetric_matrix_rank1_update: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_rank1_update_hpx
#  "hermitian_matrix_rank1_update: hpx impl")
#
#linalg_add_test_hpx(
#  symmetric_matrix_rank2_update_hpx
#  "symmetric_matrix_rank2_update: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_rank2_update_hpx
#  "hermitian_matrix_rank2_update: hpx impl")

#
# blas3 (according to P1673)
#

# matrix_product
#linalg_add_test_hpx(
#  gemm_C_AB
#  "gemm_C_AB_product: hpx impl")
#linalg_add_test_hpx(
#  gemm_C_ABT
#  "gemm_C_ABT_product: hpx impl")
#linalg_add_test_hpx(
#  gemm_C_ATB
#  "gemm_C_ATB_product: hpx impl")
#
#linalg_add_test_hpx(
#  triangular_matrix_left_product_hpx
#  "triangular_matrix_left_product_hpx: hpx impl")
#linalg_add_test_hpx(
#  triangular_matrix_right_product_hpx
#  "triangular_matrix_right_product_hpx: hpx impl")

# {symmetric,hermitian}_matrix_{left}
#linalg_add_test_hpx(
#  symmetric_matrix_left_product_hpx
#  "symmetric_matrix_left_product: hpx impl")
#linalg_add_test_hpx(
#  symmetric_matrix_right_product_hpx
#  "symmetric_matrix_right_product: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_left_product_hpx
#  "hermitian_matrix_left_product: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_right_product_hpx
#  "hermitian_matrix_right_product: hpx impl")
#
#linalg_add_test_hpx(
#  triangular_matrix_matrix_left_solve
#  "triangular_matrix_matrix_left_solve: hpx impl")
#linalg_add_test_hpx(
#  triangular_matrix_matrix_right_solve
#  "triangular_matrix_matrix_right_solve: hpx impl")
#
#linalg_add_test_hpx(
#  symmetric_matrix_rank_2k_update_hpx
#  "symmetric_matrix_rank_2k_update: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_rank_2k_update_hpx
#  "hermitian_matrix_rank_2k_update: hpx impl")
#
#linalg_add_test_hpx(
#  symmetric_matrix_rank_k_update_hpx
#  "symmetric_matrix_rank_k_update: hpx impl")
#linalg_add_test_hpx(
#  hermitian_matrix_rank_k_update_hpx
#  "hermitian_matrix_rank_k_update: hpx impl")