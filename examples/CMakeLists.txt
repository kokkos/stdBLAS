
function(linalg_add_example EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
  target_link_libraries(${EXENAME} linalg)
endfunction(linalg_add_example)

add_subdirectory(hello_world)
