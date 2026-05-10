#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static inline uint32_t micros() {
  return static_cast<uint32_t>(esp_timer_get_time());
}

#include "babyMambaEngine.h"
#include "babyMambaEngineWorker.h"

namespace {

constexpr const char* kTag = "BABYMAMBA_ESP32";
constexpr int kIterations = 10;

struct CiWorkerJob {
  const float* input;
  int channel_start;
  int channel_end;
  float accum[BABYMAMBA_D_MODEL];
  TaskHandle_t notify_target;
};

static CiWorkerJob g_ci_worker_job{};
static TaskHandle_t g_ci_worker_handle = nullptr;

struct RunSummary {
  uint32_t latency_us[kIterations];
  float last_logits[BABYMAMBA_NUM_CLASSES];
  float avg_latency_ms;
  float parity_vs_pytorch;
  float parity_vs_engine;
  float max_abs_err_vs_pytorch;
  int predicted_class;
};

float meanAbs(const float* values, int count) {
  float acc = 0.0f;
  for (int i = 0; i < count; ++i) {
    acc += fabsf(values[i]);
  }
  return acc / static_cast<float>(count);
}

float parityPercent(const float* lhs, const float* rhs, int count) {
  float diff = 0.0f;
  float denom = 0.0f;
  for (int i = 0; i < count; ++i) {
    diff += fabsf(lhs[i] - rhs[i]);
    denom += fabsf(rhs[i]);
  }
  diff /= static_cast<float>(count);
  denom /= static_cast<float>(count);
  return 100.0f * (1.0f - (diff / (denom + 1e-6f)));
}

float maxAbsErr(const float* lhs, const float* rhs, int count) {
  float value = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float err = fabsf(lhs[i] - rhs[i]);
    if (err > value) {
      value = err;
    }
  }
  return value;
}

void printChipInfo() {
  esp_chip_info_t info{};
  esp_chip_info(&info);
  ESP_LOGI(
      kTag,
      "Chip model=%d cores=%d revision=%d flash_embedded=%s psram=%s",
      static_cast<int>(info.model),
      info.cores,
      info.revision,
      (info.features & CHIP_FEATURE_EMB_FLASH) ? "yes" : "no",
      (info.features & CHIP_FEATURE_EMB_PSRAM) ? "yes" : "no");
}

void ciWorkerTask(void*) {
  while (true) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    babymamba_worker::runChannelRange(
        g_ci_worker_job.input,
        g_ci_worker_job.channel_start,
        g_ci_worker_job.channel_end,
        g_ci_worker_job.accum);
    xTaskNotifyGive(g_ci_worker_job.notify_target);
  }
}

void ensureCiWorkerTask() {
  if (g_ci_worker_handle != nullptr) {
    return;
  }
  xTaskCreatePinnedToCore(
      ciWorkerTask,
      "babymamba_ci_worker",
      16384,
      nullptr,
      5,
      &g_ci_worker_handle,
      0);
}

void runInference(const float* input_t_c, babymamba::InferenceResult* result) {
#if BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT
  ensureCiWorkerTask();
  constexpr int split = BABYMAMBA_IN_CHANNELS / 2;
  float main_accum[BABYMAMBA_D_MODEL];
  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    g_ci_worker_job.accum[d] = 0.0f;
  }
  g_ci_worker_job.input = input_t_c;
  g_ci_worker_job.channel_start = 0;
  g_ci_worker_job.channel_end = split;
  g_ci_worker_job.notify_target = xTaskGetCurrentTaskHandle();

  const uint32_t start = micros();
  xTaskNotifyGive(g_ci_worker_handle);
  babymamba::runChannelRange(input_t_c, split, BABYMAMBA_IN_CHANNELS, main_accum);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    main_accum[d] += g_ci_worker_job.accum[d];
  }
  babymamba::finishChannelIndependentFromAccum(main_accum, BABYMAMBA_IN_CHANNELS, result->logits);
  result->latency_us = micros() - start;
  result->top1 = babymamba::argmax(result->logits, BABYMAMBA_NUM_CLASSES);
#else
  babymamba::run(input_t_c, result);
#endif
}

void runBenchmarkTask(void*) {
  printChipInfo();

  multi_heap_info_t before_info{};
  multi_heap_info_t after_info{};
  heap_caps_get_info(&before_info, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  printf("=== BABYMAMBA ESP32 START ===\n");
  printf("variant=%s\n", BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT ? "ci_babymamba" : "crossover_bidir");
  printf("dataset=%s\n", BABYMAMBA_DATASET_NAME);
  printf("display_name=%s\n", BABYMAMBA_DISPLAY_NAME);
  printf("seed=%d\n", BABYMAMBA_SEED);
  printf("seq_len=%d\n", BABYMAMBA_SEQ_LEN);
  printf("in_channels=%d\n", BABYMAMBA_IN_CHANNELS);
  printf("num_classes=%d\n", BABYMAMBA_NUM_CLASSES);
#if BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT
  printf("scratch_bytes=%u\n", static_cast<unsigned>(babymamba::scratchBytes() + babymamba_worker::scratchBytes()));
#else
  printf("scratch_bytes=%u\n", static_cast<unsigned>(babymamba::scratchBytes()));
#endif
  printf("heap_total_before=%u\n", static_cast<unsigned>(before_info.total_free_bytes + before_info.total_allocated_bytes));
  printf("heap_free_before=%u\n", static_cast<unsigned>(before_info.total_free_bytes));
  printf("largest_block_before=%u\n", static_cast<unsigned>(before_info.largest_free_block));

  RunSummary summary{};
  babymamba::InferenceResult result{};

  for (int iter = 0; iter < kIterations; ++iter) {
    runInference(kFixtureInput, &result);
    summary.latency_us[iter] = result.latency_us;
    memcpy(summary.last_logits, result.logits, sizeof(summary.last_logits));
    printf("iter_%02d_latency_us=%u\n", iter + 1, static_cast<unsigned>(result.latency_us));
  }

  heap_caps_get_info(&after_info, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  uint64_t latency_sum = 0;
  for (int iter = 0; iter < kIterations; ++iter) {
    latency_sum += summary.latency_us[iter];
  }

  summary.avg_latency_ms = static_cast<float>(latency_sum) / static_cast<float>(kIterations) / 1000.0f;
  summary.parity_vs_pytorch = parityPercent(summary.last_logits, kFixturePyTorchLogits, BABYMAMBA_NUM_CLASSES);
  summary.parity_vs_engine = parityPercent(summary.last_logits, kFixtureEngineLogits, BABYMAMBA_NUM_CLASSES);
  summary.max_abs_err_vs_pytorch = maxAbsErr(summary.last_logits, kFixturePyTorchLogits, BABYMAMBA_NUM_CLASSES);
  summary.predicted_class = result.top1;

  printf("avg_latency_ms=%.6f\n", summary.avg_latency_ms);
  printf("parity_vs_pytorch_pct=%.6f\n", summary.parity_vs_pytorch);
  printf("parity_vs_engine_pct=%.6f\n", summary.parity_vs_engine);
  printf("max_abs_err_vs_pytorch=%.6f\n", summary.max_abs_err_vs_pytorch);
  printf("predicted_class=%d\n", summary.predicted_class);
  printf("predicted_label=%s\n", kBabyMambaClassNames[summary.predicted_class]);
  printf("expected_label=%s\n", kBabyMambaClassNames[BABYMAMBA_FIXTURE_LABEL]);
  printf("heap_total_after=%u\n", static_cast<unsigned>(after_info.total_free_bytes + after_info.total_allocated_bytes));
  printf("heap_free_after=%u\n", static_cast<unsigned>(after_info.total_free_bytes));
  printf("heap_used_after=%u\n", static_cast<unsigned>(after_info.total_allocated_bytes));
  printf("largest_block_after=%u\n", static_cast<unsigned>(after_info.largest_free_block));

  for (int i = 0; i < BABYMAMBA_NUM_CLASSES; ++i) {
    printf("logit_%02d=%.9f\n", i, summary.last_logits[i]);
  }

  printf("=== DONE ===\n");
  fflush(stdout);

  vTaskDelay(pdMS_TO_TICKS(2000));
  esp_restart();
}

}  // namespace

extern "C" void app_main(void) {
  xTaskCreatePinnedToCore(
      runBenchmarkTask,
      "babymamba_task",
      32768,
      nullptr,
      5,
      nullptr,
      1);
}
