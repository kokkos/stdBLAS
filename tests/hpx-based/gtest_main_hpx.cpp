//  Copyright (c) 2022 Hartmut Kaiser

#include <hpx/hpx_init.hpp>

#include <gtest/gtest.h>
#include <iostream>

#if defined(HPX_STDBLAS_ENABLE_TESTS)
namespace HPXKernelsSTD { namespace Impl {

void signal_hpx_impl_called(std::string_view functionName)
{
    std::cout << functionName << ": hpx impl" << std::endl;
}

}}    // namespace HPXKernelsSTD::Impl
#endif

int hpx_main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    int err = RUN_ALL_TESTS();
    hpx::finalize();
    return err;
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
