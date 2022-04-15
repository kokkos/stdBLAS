
macro(linalg_add_test_kokkos TESTNAME FINDSTR)

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
  add_executable(${testExe} ${TESTNAME}.cpp gtest_main_kokkos.cpp)
  target_link_libraries(${testExe} linalg GTest::GTest)

  if (NOT compileDefs STREQUAL "")
    target_compile_definitions(${testExe} PRIVATE -D${compileDefs} -DKOKKOS_STDBLAS_ENABLE_TESTS)
  else()
    target_compile_definitions(${testExe} PRIVATE -DKOKKOS_STDBLAS_ENABLE_TESTS)
  endif()

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

# add tests below

# mdspan_to_view is simple enough we don't need to detect any
# output from the impl itself so just add basic test
set(testExe mdspan_to_view)
add_executable(${testExe} mdspan_to_view.cpp gtest_main_kokkos.cpp)
target_link_libraries(${testExe} linalg GTest::GTest)
add_test(NAME utest_${testExe} COMMAND ${testExe})
set_tests_properties(utest_${testExe} PROPERTIES FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;FAILED")

linalg_add_test_kokkos(
  matrix_rank1_update_kokkos
  "matrix_rank1_update: kokkos impl")

linalg_add_test_kokkos(
  dot_kokkos
  "dot: kokkos impl")
linalg_add_test_kokkos(
  dotc_kokkos
  "dot: kokkos impl") # this his not a typo, dotc calls dot underneath

linalg_add_test_kokkos(
  swap_elements_rank1_kokkos
  "swap_elements: kokkos impl")
linalg_add_test_kokkos(
  swap_elements_rank2_kokkos
  "swap_elements: kokkos impl")
linalg_add_test_kokkos(
  add_kokkos
  "add: kokkos impl")
linalg_add_test_kokkos(
  scale_rank1_kokkos
  "scale: kokkos impl")
linalg_add_test_kokkos(
  scale_rank2_kokkos
  "scale: kokkos impl")
