
function(linalg_add_example EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
  target_link_libraries(${EXENAME} linalg)
endfunction(linalg_add_example)

linalg_add_example(01_scale)
linalg_add_example(02_matrix_vector_product_basic)
linalg_add_example(03_matrix_vector_product_mixedprec)

if(LINALG_ENABLE_KOKKOS)
  add_subdirectory(kokkos-based)
endif()
