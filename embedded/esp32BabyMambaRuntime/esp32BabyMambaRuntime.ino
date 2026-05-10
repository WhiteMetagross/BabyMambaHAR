#include <Arduino.h>
#include <math.h>
#include "babyMambaEngine.h"

namespace {

float meanFloat(const uint32_t* values, size_t count) {
  if (count == 0) {
    return 0.0f;
  }
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += static_cast<double>(values[i]);
  }
  return static_cast<float>(sum / static_cast<double>(count));
}

float computeParityPercent(const float* a, const float* b, size_t count) {
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < count; ++i) {
    numerator += fabs(static_cast<double>(a[i] - b[i]));
    denominator += fabs(static_cast<double>(b[i]));
  }
  return 100.0f * (1.0f - static_cast<float>(numerator / (denominator + 1e-6)));
}

float computeMaxAbsError(const float* a, const float* b, size_t count) {
  float max_error = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    max_error = fmaxf(max_error, fabsf(a[i] - b[i]));
  }
  return max_error;
}

void printLogits(const char* prefix, const float* logits) {
  Serial.print(prefix);
  Serial.print("=");
  for (size_t i = 0; i < BABYMAMBA_NUM_CLASSES; ++i) {
    if (i != 0) {
      Serial.print(",");
    }
    Serial.print(logits[i], 6);
  }
  Serial.println();
}

}  // namespace

void setup() {
  Serial.begin(115200);
  const uint32_t serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start) < 10000) {
    delay(10);
  }
  delay(4000);

  Serial.println("=== BABYMAMBA_ESP32_START ===");
  Serial.print("dataset_key=");
  Serial.println(BABYMAMBA_DATASET_NAME);
  Serial.print("dataset_display=");
  Serial.println(BABYMAMBA_DISPLAY_NAME);
  Serial.print("variant_name=");
#if BABYMAMBA_VARIANT_CROSSOVER
  Serial.println("crossoverBiDirBabyMambaHar");
#else
  Serial.println("ciBabyMambaHar");
#endif
  Serial.print("model_seed=");
  Serial.println(BABYMAMBA_SEED);
  Serial.print("scratch_bytes=");
  Serial.println(babymamba::scratchBytes());
  Serial.print("input_bytes=");
  Serial.println(sizeof(kFixtureInput));
  Serial.print("reference_bytes=");
  Serial.println(sizeof(kFixturePyTorchLogits) + sizeof(kFixtureEngineLogits));
  Serial.println("stage=warmup_begin");

  babymamba::InferenceResult warmup{};
  babymamba::run(kFixtureInput, &warmup);
  Serial.print("warmup_latency_us=");
  Serial.println(warmup.latency_us);
  Serial.println("stage=warmup_done");

  constexpr size_t kRuns = 10;
  uint32_t latencies[kRuns] = {};
  babymamba::InferenceResult result{};
  for (size_t run_idx = 0; run_idx < kRuns; ++run_idx) {
    Serial.print("stage=run_begin_");
    Serial.println(static_cast<int>(run_idx));
    babymamba::run(kFixtureInput, &result);
    latencies[run_idx] = result.latency_us;
    Serial.print("run_latency_us=");
    Serial.println(result.latency_us);
  }

  const float parity_vs_pytorch = computeParityPercent(
      result.logits, kFixturePyTorchLogits, BABYMAMBA_NUM_CLASSES);
  const float parity_vs_engine = computeParityPercent(
      result.logits, kFixtureEngineLogits, BABYMAMBA_NUM_CLASSES);
  const float max_abs_pytorch = computeMaxAbsError(
      result.logits, kFixturePyTorchLogits, BABYMAMBA_NUM_CLASSES);

  Serial.print("avg_latency_us=");
  Serial.println(meanFloat(latencies, kRuns), 3);
  Serial.print("predicted_class=");
  Serial.println(result.top1);
  Serial.print("predicted_label=");
  Serial.println(kBabyMambaClassNames[result.top1]);
  Serial.print("expected_label_index=");
  Serial.println(BABYMAMBA_FIXTURE_LABEL);
  Serial.print("expected_label_name=");
  Serial.println(kBabyMambaClassNames[BABYMAMBA_FIXTURE_LABEL]);
  Serial.print("parity_vs_pytorch_pct=");
  Serial.println(parity_vs_pytorch, 6);
  Serial.print("parity_vs_export_engine_pct=");
  Serial.println(parity_vs_engine, 6);
  Serial.print("max_abs_error_vs_pytorch=");
  Serial.println(max_abs_pytorch, 6);
  printLogits("final_logits", result.logits);
  Serial.println("=== BABYMAMBA_PICO2_DONE ===");
}

void loop() {}

