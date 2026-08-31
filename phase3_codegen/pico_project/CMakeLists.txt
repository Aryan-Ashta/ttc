cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(ttc_phase3 C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(ttc_phase3
    pico_main.c
    model.c
)

target_link_libraries(ttc_phase3 pico_stdlib)

# UART stdio only -- deliberately no USB stdio, so this build doesn't
# need the tinyusb submodule at all.
pico_enable_stdio_uart(ttc_phase3 1)
pico_enable_stdio_usb(ttc_phase3 0)

pico_add_extra_outputs(ttc_phase3)
