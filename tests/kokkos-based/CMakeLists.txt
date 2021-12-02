
macro(linalg_add_test_kokkos TESTNAME TESTSRCS)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} gtest_main_kokkos.cpp)
  target_link_libraries(${testNameFinal} linalg GTest::GTest)
  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()

linalg_add_test_kokkos(utest_add_kokkos add_kokkos.cpp)
linalg_add_test_kokkos(utest_dot_kokkos dot_kokkos.cpp)
