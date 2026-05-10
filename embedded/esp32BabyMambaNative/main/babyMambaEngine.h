#pragma once

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "babyMambaWeights.h"

namespace babymamba {

#ifndef BABYMAMBA_USE_INT8_PROJECTIONS
#define BABYMAMBA_USE_INT8_PROJECTIONS 0
#endif

#if BABYMAMBA_USE_INT8_PROJECTIONS
using InProjRow = int8_t[BABYMAMBA_D_MODEL];
using XProjRow = int8_t[BABYMAMBA_D_INNER];
using DtProjWeightRow = int8_t[BABYMAMBA_DT_RANK];
using OutProjRow = int8_t[BABYMAMBA_D_INNER];
#else
using InProjRow = float[BABYMAMBA_D_MODEL];
using XProjRow = float[BABYMAMBA_D_INNER];
using DtProjWeightRow = float[BABYMAMBA_DT_RANK];
using OutProjRow = float[BABYMAMBA_D_INNER];
#endif

#if BABYMAMBA_USE_INT8_PROJECTIONS
#define BABYMAMBA_PATCH_POINTWISE_WEIGHTS kPatchPointwiseQ
#define BABYMAMBA_PATCH_POINTWISE_SCALES kPatchPointwiseScale
#define BABYMAMBA_GATED_PROJECTION_WEIGHTS kGatedProjectionWeightQ
#define BABYMAMBA_GATED_PROJECTION_SCALES kGatedProjectionWeightScale
#define BABYMAMBA_HEAD_LINEAR_WEIGHTS kHeadLinearWeightQ
#define BABYMAMBA_HEAD_LINEAR_SCALES kHeadLinearWeightScale
#else
#define BABYMAMBA_PATCH_POINTWISE_WEIGHTS kPatchPointwise
#define BABYMAMBA_PATCH_POINTWISE_SCALES nullptr
#define BABYMAMBA_GATED_PROJECTION_WEIGHTS kGatedProjectionWeight
#define BABYMAMBA_GATED_PROJECTION_SCALES nullptr
#define BABYMAMBA_HEAD_LINEAR_WEIGHTS kHeadLinearWeight
#define BABYMAMBA_HEAD_LINEAR_SCALES nullptr
#endif

struct InferenceResult {
  float logits[BABYMAMBA_NUM_CLASSES];
  uint32_t latency_us;
  int top1;
};

struct LayerParams {
  const float* pre_norm_weight;
  const float* pre_norm_bias;
  const float* post_norm_weight;
  const float* post_norm_bias;
  const float (*a_log)[BABYMAMBA_D_STATE];
  const float* d_skip;
  const InProjRow* in_proj;
  const float* in_proj_scale;
  const float (*conv1d_weight)[BABYMAMBA_D_CONV];
  const float* conv1d_bias;
  const XProjRow* x_proj;
  const float* x_proj_scale;
  const DtProjWeightRow* dt_proj_weight;
  const float* dt_proj_weight_scale;
  const float* dt_proj_bias;
  const OutProjRow* out_proj;
  const float* out_proj_scale;
};

struct Scratch {
  float stem[BABYMAMBA_D_MODEL][BABYMAMBA_SEQ_LEN];
  float seq[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL];
  float norm[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL];
  float fwd[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL];
  float bwd[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL];
  float hidden[BABYMAMBA_D_INNER][BABYMAMBA_D_STATE];
  float cum_log_a[BABYMAMBA_D_INNER][BABYMAMBA_D_STATE];
  float cum_scaled[BABYMAMBA_D_INNER][BABYMAMBA_D_STATE];
  float history[BABYMAMBA_D_INNER][BABYMAMBA_D_CONV];
  float x_part[BABYMAMBA_D_INNER];
  float z_part[BABYMAMBA_D_INNER];
  float conv[BABYMAMBA_D_INNER];
  float x_act[BABYMAMBA_D_INNER];
  float x_proj_full[BABYMAMBA_DT_RANK + 2 * BABYMAMBA_D_STATE];
  float dt[BABYMAMBA_D_INNER];
  float y_inner[BABYMAMBA_D_INNER];
  float pooled[BABYMAMBA_D_MODEL];
  float combined[BABYMAMBA_D_MODEL];
  float head_norm[BABYMAMBA_D_MODEL];
  float attention_scores[BABYMAMBA_PATCH_OUT_LEN];
  float channel_accum[BABYMAMBA_D_MODEL];
};

static Scratch g_scratch;

constexpr int kExpLutSize = 257;
constexpr float kExpLutMin = -16.0f;
constexpr float kExpLutMax = 8.0f;
constexpr float kExpLutStep = (kExpLutMax - kExpLutMin) / static_cast<float>(kExpLutSize - 1);
static float g_exp_lut[kExpLutSize];
static bool g_exp_lut_ready = false;

static inline void ensureExpLut() {
  if (g_exp_lut_ready) {
    return;
  }
  for (int i = 0; i < kExpLutSize; ++i) {
    const float x = kExpLutMin + (static_cast<float>(i) * kExpLutStep);
    g_exp_lut[i] = expf(x);
  }
  g_exp_lut_ready = true;
}

static inline float fastExp(float x) {
  ensureExpLut();
  if (x <= kExpLutMin) {
    return expf(x);
  }
  if (x >= kExpLutMax) {
    return expf(x);
  }
  const float position = (x - kExpLutMin) / kExpLutStep;
  const int index = static_cast<int>(position);
  const float fraction = position - static_cast<float>(index);
  const float left = g_exp_lut[index];
  const float right = g_exp_lut[index + 1];
  return left + (right - left) * fraction;
}

static inline float sigmoid(float x) {
  return 1.0f / (1.0f + fastExp(-x));
}

static inline float silu(float x) {
  return x * sigmoid(x);
}

static inline float softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return fastExp(x);
  }
  return log1pf(fastExp(x));
}

static inline float clampf(float x, float lo, float hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static inline uint32_t scratchBytes() {
  return static_cast<uint32_t>(sizeof(g_scratch));
}

static inline float dotProduct(const float* weights, const float* values, int count) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  int i = 0;
  for (; i + 3 < count; i += 4) {
    acc0 = fmaf(weights[i], values[i], acc0);
    acc1 = fmaf(weights[i + 1], values[i + 1], acc1);
    acc2 = fmaf(weights[i + 2], values[i + 2], acc2);
    acc3 = fmaf(weights[i + 3], values[i + 3], acc3);
  }
  float acc = (acc0 + acc1) + (acc2 + acc3);
  for (; i < count; ++i) {
    acc = fmaf(weights[i], values[i], acc);
  }
  return acc;
}

static inline float dotProductQ8(const int8_t* weights, float scale, const float* values, int count) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  int i = 0;
  for (; i + 3 < count; i += 4) {
    acc0 = fmaf(static_cast<float>(weights[i]), values[i], acc0);
    acc1 = fmaf(static_cast<float>(weights[i + 1]), values[i + 1], acc1);
    acc2 = fmaf(static_cast<float>(weights[i + 2]), values[i + 2], acc2);
    acc3 = fmaf(static_cast<float>(weights[i + 3]), values[i + 3], acc3);
  }
  float acc = (acc0 + acc1) + (acc2 + acc3);
  for (; i < count; ++i) {
    acc = fmaf(static_cast<float>(weights[i]), values[i], acc);
  }
  return acc * scale;
}

#if BABYMAMBA_USE_INT8_PROJECTIONS
#define BABYMAMBA_PROJECT_ROW(matrix, scales, index, values, count) \
  dotProductQ8((matrix)[index], (scales)[index], values, count)
#else
#define BABYMAMBA_PROJECT_ROW(matrix, scales, index, values, count) \
  dotProduct((matrix)[index], values, count)
#endif

static inline const LayerParams getLayerParams(int index) {
#if BABYMAMBA_USE_INT8_PROJECTIONS
  switch (index) {
    case 0:
      return {
          kLayer0PreNormWeight, kLayer0PreNormBias, kLayer0PostNormWeight, kLayer0PostNormBias,
          kLayer0ALog, kLayer0D, kLayer0InProjQ, kLayer0InProjScale, kLayer0Conv1dWeight, kLayer0Conv1dBias,
          kLayer0XProjQ, kLayer0XProjScale, kLayer0DtProjWeightQ, kLayer0DtProjWeightScale, kLayer0DtProjBias,
          kLayer0OutProjQ, kLayer0OutProjScale,
      };
    case 1:
      return {
          kLayer1PreNormWeight, kLayer1PreNormBias, kLayer1PostNormWeight, kLayer1PostNormBias,
          kLayer1ALog, kLayer1D, kLayer1InProjQ, kLayer1InProjScale, kLayer1Conv1dWeight, kLayer1Conv1dBias,
          kLayer1XProjQ, kLayer1XProjScale, kLayer1DtProjWeightQ, kLayer1DtProjWeightScale, kLayer1DtProjBias,
          kLayer1OutProjQ, kLayer1OutProjScale,
      };
    case 2:
      return {
          kLayer2PreNormWeight, kLayer2PreNormBias, kLayer2PostNormWeight, kLayer2PostNormBias,
          kLayer2ALog, kLayer2D, kLayer2InProjQ, kLayer2InProjScale, kLayer2Conv1dWeight, kLayer2Conv1dBias,
          kLayer2XProjQ, kLayer2XProjScale, kLayer2DtProjWeightQ, kLayer2DtProjWeightScale, kLayer2DtProjBias,
          kLayer2OutProjQ, kLayer2OutProjScale,
      };
    default:
      return {
          kLayer3PreNormWeight, kLayer3PreNormBias, kLayer3PostNormWeight, kLayer3PostNormBias,
          kLayer3ALog, kLayer3D, kLayer3InProjQ, kLayer3InProjScale, kLayer3Conv1dWeight, kLayer3Conv1dBias,
          kLayer3XProjQ, kLayer3XProjScale, kLayer3DtProjWeightQ, kLayer3DtProjWeightScale, kLayer3DtProjBias,
          kLayer3OutProjQ, kLayer3OutProjScale,
      };
  }
#else
  switch (index) {
    case 0:
      return {
          kLayer0PreNormWeight, kLayer0PreNormBias, kLayer0PostNormWeight, kLayer0PostNormBias,
          kLayer0ALog, kLayer0D, kLayer0InProj, nullptr, kLayer0Conv1dWeight, kLayer0Conv1dBias,
          kLayer0XProj, nullptr, kLayer0DtProjWeight, nullptr, kLayer0DtProjBias, kLayer0OutProj, nullptr,
      };
    case 1:
      return {
          kLayer1PreNormWeight, kLayer1PreNormBias, kLayer1PostNormWeight, kLayer1PostNormBias,
          kLayer1ALog, kLayer1D, kLayer1InProj, nullptr, kLayer1Conv1dWeight, kLayer1Conv1dBias,
          kLayer1XProj, nullptr, kLayer1DtProjWeight, nullptr, kLayer1DtProjBias, kLayer1OutProj, nullptr,
      };
    case 2:
      return {
          kLayer2PreNormWeight, kLayer2PreNormBias, kLayer2PostNormWeight, kLayer2PostNormBias,
          kLayer2ALog, kLayer2D, kLayer2InProj, nullptr, kLayer2Conv1dWeight, kLayer2Conv1dBias,
          kLayer2XProj, nullptr, kLayer2DtProjWeight, nullptr, kLayer2DtProjBias, kLayer2OutProj, nullptr,
      };
    default:
      return {
          kLayer3PreNormWeight, kLayer3PreNormBias, kLayer3PostNormWeight, kLayer3PostNormBias,
          kLayer3ALog, kLayer3D, kLayer3InProj, nullptr, kLayer3Conv1dWeight, kLayer3Conv1dBias,
          kLayer3XProj, nullptr, kLayer3DtProjWeight, nullptr, kLayer3DtProjBias, kLayer3OutProj, nullptr,
      };
  }
#endif
}

static inline void layerNorm1d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int count) {
  float mean = 0.0f;
  for (int i = 0; i < count; ++i) {
    mean += input[i];
  }
  mean /= static_cast<float>(count);

  float var = 0.0f;
  for (int i = 0; i < count; ++i) {
    float diff = input[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(count);
  const float inv_std = 1.0f / sqrtf(var + 1e-5f);

  for (int i = 0; i < count; ++i) {
    output[i] = ((input[i] - mean) * inv_std) * weight[i] + bias[i];
  }
}

static inline int argmax(const float* values, int count) {
  int best = 0;
  float best_value = values[0];
  for (int i = 1; i < count; ++i) {
    if (values[i] > best_value) {
      best_value = values[i];
      best = i;
    }
  }
  return best;
}

static inline void stemForwardCrossover(const float* input_t_c) {
  for (int oc = 0; oc < BABYMAMBA_D_MODEL; ++oc) {
    for (int t = 0; t < BABYMAMBA_SEQ_LEN; ++t) {
      float acc = kStemBias[oc];
      for (int ic = 0; ic < BABYMAMBA_IN_CHANNELS; ++ic) {
        for (int k = 0; k < BABYMAMBA_STEM_KERNEL; ++k) {
          int src_t = t + k - (BABYMAMBA_STEM_KERNEL / 2);
          if (src_t < 0 || src_t >= BABYMAMBA_SEQ_LEN) {
            continue;
          }
          const float input_value = input_t_c[src_t * BABYMAMBA_IN_CHANNELS + ic];
          acc += input_value * kStemWeight[oc][ic][k];
        }
      }
      g_scratch.stem[oc][t] = silu(acc);
    }
  }
}

static inline void stemForwardCiChannel(const float* input_t_c, int channel_idx) {
  for (int oc = 0; oc < BABYMAMBA_D_MODEL; ++oc) {
    for (int t = 0; t < BABYMAMBA_SEQ_LEN; ++t) {
      float acc = kStemBias[oc];
      for (int k = 0; k < BABYMAMBA_STEM_KERNEL; ++k) {
        int src_t = t + k - (BABYMAMBA_STEM_KERNEL / 2);
        if (src_t < 0 || src_t >= BABYMAMBA_SEQ_LEN) {
          continue;
        }
        const float input_value = input_t_c[src_t * BABYMAMBA_IN_CHANNELS + channel_idx];
        acc += input_value * kStemWeight[oc][0][k];
      }
      g_scratch.stem[oc][t] = silu(acc);
    }
  }
}

static inline void patchForward() {
  for (int c = 0; c < BABYMAMBA_D_MODEL; ++c) {
    for (int out_idx = 0; out_idx < BABYMAMBA_PATCH_OUT_LEN; ++out_idx) {
      const int start = out_idx * BABYMAMBA_PATCH_STRIDE - (BABYMAMBA_PATCH_KERNEL / 4);
      float acc = 0.0f;
      for (int k = 0; k < BABYMAMBA_PATCH_KERNEL; ++k) {
        const int src = start + k;
        if (src < 0 || src >= BABYMAMBA_SEQ_LEN) {
          continue;
        }
        acc += g_scratch.stem[c][src] * kPatchDepthwise[c][k];
      }
      g_scratch.norm[out_idx][c] = acc;
    }
  }

  for (int out_idx = 0; out_idx < BABYMAMBA_PATCH_OUT_LEN; ++out_idx) {
    for (int oc = 0; oc < BABYMAMBA_D_MODEL; ++oc) {
      const float acc = kPatchBias[oc] + BABYMAMBA_PROJECT_ROW(
          BABYMAMBA_PATCH_POINTWISE_WEIGHTS,
          BABYMAMBA_PATCH_POINTWISE_SCALES,
          oc,
          g_scratch.norm[out_idx],
          BABYMAMBA_D_MODEL);
      g_scratch.seq[out_idx][oc] = silu(acc) + kPosEmbed[0][out_idx][oc];
    }
  }
}

static inline void selectiveScanDirection(
    const float input[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL],
    const LayerParams& layer,
    bool reverse,
    float output[BABYMAMBA_PATCH_OUT_LEN][BABYMAMBA_D_MODEL]) {
  memset(g_scratch.hidden, 0, sizeof(g_scratch.hidden));
#if BABYMAMBA_SCAN_IMPL_FALLBACK
  memset(g_scratch.cum_log_a, 0, sizeof(g_scratch.cum_log_a));
  memset(g_scratch.cum_scaled, 0, sizeof(g_scratch.cum_scaled));
#endif
  memset(g_scratch.history, 0, sizeof(g_scratch.history));

  for (int step = 0; step < BABYMAMBA_PATCH_OUT_LEN; ++step) {
    const int idx = reverse ? (BABYMAMBA_PATCH_OUT_LEN - 1 - step) : step;
    const float* token = input[idx];

    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      g_scratch.x_part[i] = BABYMAMBA_PROJECT_ROW(
          layer.in_proj,
          layer.in_proj_scale,
          i,
          token,
          BABYMAMBA_D_MODEL);
    }
    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      g_scratch.z_part[i] = BABYMAMBA_PROJECT_ROW(
          layer.in_proj,
          layer.in_proj_scale,
          BABYMAMBA_D_INNER + i,
          token,
          BABYMAMBA_D_MODEL);
    }

    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      for (int k = 0; k < BABYMAMBA_D_CONV - 1; ++k) {
        g_scratch.history[i][k] = g_scratch.history[i][k + 1];
      }
      g_scratch.history[i][BABYMAMBA_D_CONV - 1] = g_scratch.x_part[i];
    }

    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      float acc = layer.conv1d_bias[i];
      for (int k = 0; k < BABYMAMBA_D_CONV; ++k) {
        acc += g_scratch.history[i][k] * layer.conv1d_weight[i][k];
      }
      g_scratch.conv[i] = acc;
      g_scratch.x_act[i] = silu(acc);
    }

    for (int i = 0; i < BABYMAMBA_DT_RANK + 2 * BABYMAMBA_D_STATE; ++i) {
      g_scratch.x_proj_full[i] = BABYMAMBA_PROJECT_ROW(
          layer.x_proj,
          layer.x_proj_scale,
          i,
          g_scratch.x_act,
          BABYMAMBA_D_INNER);
    }

    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      g_scratch.dt[i] = softplus(
          layer.dt_proj_bias[i] +
          BABYMAMBA_PROJECT_ROW(
              layer.dt_proj_weight,
              layer.dt_proj_weight_scale,
              i,
              g_scratch.x_proj_full,
              BABYMAMBA_DT_RANK));
    }

    for (int i = 0; i < BABYMAMBA_D_INNER; ++i) {
      float y_val = layer.d_skip[i] * g_scratch.x_act[i];
      const float gate = silu(g_scratch.z_part[i]);
      for (int n = 0; n < BABYMAMBA_D_STATE; ++n) {
        const float B_val = g_scratch.x_proj_full[BABYMAMBA_DT_RANK + n];
        const float C_val = g_scratch.x_proj_full[BABYMAMBA_DT_RANK + BABYMAMBA_D_STATE + n];
#if BABYMAMBA_SCAN_IMPL_FALLBACK
        const float a_val = -fastExp(layer.a_log[i][n]);
        const float delta_a = fastExp(g_scratch.dt[i] * a_val);
        const float log_a = logf(fmaxf(delta_a, 1e-6f));
        g_scratch.cum_log_a[i][n] += log_a;
        const float cum_a = fastExp(clampf(g_scratch.cum_log_a[i][n], -3.4e38f, 20.0f));
        const float inv_cum_a = fastExp(-clampf(g_scratch.cum_log_a[i][n], -20.0f, 20.0f));
        const float scaled_bx = (g_scratch.dt[i] * B_val * g_scratch.x_act[i]) * inv_cum_a;
        g_scratch.cum_scaled[i][n] += scaled_bx;
        g_scratch.hidden[i][n] = cum_a * g_scratch.cum_scaled[i][n];
#else
        const float decay = fastExp(g_scratch.dt[i] * (-fastExp(layer.a_log[i][n])));
        g_scratch.hidden[i][n] =
            decay * g_scratch.hidden[i][n] + (g_scratch.dt[i] * B_val * g_scratch.x_act[i]);
#endif
        y_val += g_scratch.hidden[i][n] * C_val;
      }
      g_scratch.y_inner[i] = y_val * gate;
    }

    for (int out_dim = 0; out_dim < BABYMAMBA_D_MODEL; ++out_dim) {
      output[idx][out_dim] = BABYMAMBA_PROJECT_ROW(
          layer.out_proj,
          layer.out_proj_scale,
          out_dim,
          g_scratch.y_inner,
          BABYMAMBA_D_INNER);
    }
  }
}

static inline void runLayer(const LayerParams& layer) {
  for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
    layerNorm1d(
        g_scratch.seq[t],
        layer.pre_norm_weight,
        layer.pre_norm_bias,
        g_scratch.norm[t],
        BABYMAMBA_D_MODEL);
  }

  selectiveScanDirection(g_scratch.norm, layer, false, g_scratch.fwd);
  selectiveScanDirection(g_scratch.norm, layer, true, g_scratch.bwd);

  for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
    for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
      g_scratch.combined[d] = g_scratch.seq[t][d] + g_scratch.fwd[t][d] + g_scratch.bwd[t][d];
    }
    layerNorm1d(
        g_scratch.combined,
        layer.post_norm_weight,
        layer.post_norm_bias,
        g_scratch.seq[t],
        BABYMAMBA_D_MODEL);
  }
}

static inline void meanPoolSequence() {
  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    float acc = 0.0f;
    for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
      acc += g_scratch.seq[t][d];
    }
    g_scratch.pooled[d] = acc / static_cast<float>(BABYMAMBA_PATCH_OUT_LEN);
  }
}

#if BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT
static inline void gatedAttentionPool() {
  float max_score = -3.4e38f;
  for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
    float score = 0.0f;
    for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
      const float acc = kGatedProjectionBias[d] + BABYMAMBA_PROJECT_ROW(
          BABYMAMBA_GATED_PROJECTION_WEIGHTS,
          BABYMAMBA_GATED_PROJECTION_SCALES,
          d,
          g_scratch.seq[t],
          BABYMAMBA_D_MODEL);
      g_scratch.combined[d] = tanhf(acc);
      score += g_scratch.combined[d] * kGatedContext[d];
    }
    g_scratch.attention_scores[t] = score;
    if (score > max_score) {
      max_score = score;
    }
  }

  float score_sum = 0.0f;
  for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
    const float exp_score = fastExp(g_scratch.attention_scores[t] - max_score);
    g_scratch.attention_scores[t] = exp_score;
    score_sum += exp_score;
  }
  const float inv_score_sum = 1.0f / (score_sum + 1e-6f);

  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    float acc = 0.0f;
    for (int t = 0; t < BABYMAMBA_PATCH_OUT_LEN; ++t) {
      acc += (g_scratch.attention_scores[t] * inv_score_sum) * g_scratch.seq[t][d];
    }
    g_scratch.pooled[d] = acc;
  }
}
#else
static inline void gatedAttentionPool() {}
#endif

static inline void headFromPooled(float* logits_out) {
  layerNorm1d(
      g_scratch.pooled,
      kHeadNormWeight,
      kHeadNormBias,
      g_scratch.head_norm,
      BABYMAMBA_D_MODEL);

  for (int c = 0; c < BABYMAMBA_NUM_CLASSES; ++c) {
    logits_out[c] = kHeadLinearBias[c] + BABYMAMBA_PROJECT_ROW(
        BABYMAMBA_HEAD_LINEAR_WEIGHTS,
        BABYMAMBA_HEAD_LINEAR_SCALES,
        c,
        g_scratch.head_norm,
        BABYMAMBA_D_MODEL);
  }
}

static inline void runCrossover(const float* input_t_c, InferenceResult* result) {
  stemForwardCrossover(input_t_c);
  patchForward();
  for (int layer_idx = 0; layer_idx < BABYMAMBA_NUM_LAYERS; ++layer_idx) {
    runLayer(getLayerParams(layer_idx));
  }
  meanPoolSequence();
  headFromPooled(result->logits);
}

static inline void runChannelRange(
    const float* input_t_c,
    int channel_start,
    int channel_end,
    float* accum_out);

static inline void finishChannelIndependentFromAccum(
    const float* accum,
    int channel_count,
    float* logits_out);

static inline void runChannelIndependent(const float* input_t_c, InferenceResult* result) {
  runChannelRange(input_t_c, 0, BABYMAMBA_IN_CHANNELS, g_scratch.channel_accum);
  finishChannelIndependentFromAccum(g_scratch.channel_accum, BABYMAMBA_IN_CHANNELS, result->logits);
}

static inline void runChannelRange(
    const float* input_t_c,
    int channel_start,
    int channel_end,
    float* accum_out) {
  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    accum_out[d] = 0.0f;
  }

  for (int channel_idx = channel_start; channel_idx < channel_end; ++channel_idx) {
    stemForwardCiChannel(input_t_c, channel_idx);
    patchForward();
    for (int layer_idx = 0; layer_idx < BABYMAMBA_NUM_LAYERS; ++layer_idx) {
      runLayer(getLayerParams(layer_idx));
    }
    gatedAttentionPool();
    for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
      accum_out[d] += g_scratch.pooled[d];
    }
  }
}

static inline void finishChannelIndependentFromAccum(
    const float* accum,
    int channel_count,
    float* logits_out) {
  const float inv_channels = 1.0f / static_cast<float>(channel_count);
  for (int d = 0; d < BABYMAMBA_D_MODEL; ++d) {
    g_scratch.pooled[d] = accum[d] * inv_channels;
  }
  headFromPooled(logits_out);
}

static inline void run(const float* input_t_c, InferenceResult* result) {
  const uint32_t start = micros();
#if BABYMAMBA_VARIANT_CHANNEL_INDEPENDENT
  runChannelIndependent(input_t_c, result);
#else
  runCrossover(input_t_c, result);
#endif
  result->latency_us = micros() - start;
  result->top1 = argmax(result->logits, BABYMAMBA_NUM_CLASSES);
}

}  // namespace babymamba
