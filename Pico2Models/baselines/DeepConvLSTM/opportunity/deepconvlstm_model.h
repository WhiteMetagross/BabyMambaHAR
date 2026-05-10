#pragma once

#include <cstddef>
#include <cstdint>

// ESP32 keeps model bytes in flash via PROGMEM. RP2040/Pico stores `const`
// data in XIP flash by default, so the empty fallback is the desired behavior.
#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
#include <pgmspace.h>
#define MICROBI_MODEL_STORAGE PROGMEM
#else
#define MICROBI_MODEL_STORAGE
#endif

#if defined(__GNUC__)
#define MICROBI_MODEL_ALIGN __attribute__((aligned(16)))
#else
#define MICROBI_MODEL_ALIGN
#endif

extern MICROBI_MODEL_ALIGN const unsigned char deepconvlstm_model[] MICROBI_MODEL_STORAGE;
extern const unsigned int deepconvlstm_model_len;

extern const float deepconvlstm_model_input_scale;
extern const int deepconvlstm_model_input_zero_point;
extern const float deepconvlstm_model_output_scale;
extern const int deepconvlstm_model_output_zero_point;
