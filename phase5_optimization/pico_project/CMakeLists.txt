cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(ttc_phase5 C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(ttc_phase5
    pico_main.c
    model.c
    model_opt.c
)

target_link_libraries(ttc_phase5 pico_stdlib hardware_clocks)

pico_enable_stdio_uart(ttc_phase5 1)
pico_enable_stdio_usb(ttc_phase5 0)

pico_add_extra_outputs(ttc_phase5)
