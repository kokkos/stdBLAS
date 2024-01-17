include(FindUnixCommands)

# run test executable
execute_process(
  COMMAND ${EXE_NAME}
  RESULT_VARIABLE RES_A
  OUTPUT_FILE ${LOG_FILE})

# first check that numerically the test passes
if(RES_A)
  message(FATAL_ERROR "numerical test failed")
else()
  message("numerical test succeeded")
endif()

# just checking the numerics is not enough 
# because we need to ensure that this result 
# comes from the kokkos impl 
# so we check that the proper string is found
# which signals that the correct Kokkos impl was found/called
set(CMD "grep -R '${TEST_STRING_FIND}' ${LOG_FILE} > /dev/null")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES_B)
if(RES_B)
  message(FATAL_ERROR
    "test failed: ${ALGO_NAME} did not call the correct Kokkos impl")
else()
  message("${ALGO_NAME} called the correct Kokkos impl")
endif()
