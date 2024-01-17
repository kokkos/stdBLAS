#  Copyright (c) 2022 Hartmut Kaiser

function(linalg_add_example_hpx EXENAME)
  linalg_add_example(${EXENAME})
  if(LINALG_ENABLE_HPX)
    target_link_libraries(${EXENAME} PRIVATE HPX::wrap_main)
  endif()
  set_target_properties(${EXENAME} PROPERTIES FOLDER "Examples/HPX")
endfunction(linalg_add_example_hpx)

linalg_add_example_hpx(add_hpx)
#linalg_add_example_hpx(dot_hpx)
#linalg_add_example_hpx(dotc_hpx)
#linalg_add_example_hpx(idx_abs_max_hpx)
#linalg_add_example_hpx(vector_norm2_hpx)
#linalg_add_example_hpx(vector_abs_sum_hpx)
#linalg_add_example_hpx(vector_sum_of_squares_hpx)
linalg_add_example_hpx(scale_hpx)
#linalg_add_example_hpx(matrix_vector_product_hpx)
