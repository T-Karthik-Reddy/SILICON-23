/***************************************************************************//**
 * @file
 * @brief Audio classifier application
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include "os.h"
#include "sl_power_manager.h"
#include "sl_status.h"
#include "sl_led.h"
#include "sl_simple_led_instances.h"
#include "sl_mic.h"
#include "audio_classifier.h"
#include "app.h"
#include "config/audio_classifier_config.h"
#include "sl_tflite_micro_init.h"
#include "sl_sleeptimer.h"
#include <cmath>
#include <cstdint>
#include <cstring>

#if SL_SIMPLE_LED_COUNT < 2
  #error "Sample application requires two leds"
#endif

extern "C" {
#include "arm_math.h"
}

// Micrium OS Task variables
static OS_TCB tcb;
static CPU_STK stack[TASK_STACK_SIZE];
const char* category_labels[] = CATEGORY_LABELS;
static int category_label_count = sizeof(category_labels) / sizeof(category_labels[0]);
int category_count = 0;

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kSampleRateHz = 16000;
constexpr uint32_t kWindowMs = 30;
constexpr uint32_t kHopMs = 10;
constexpr uint32_t kNfft = 512;
constexpr uint32_t kNMels = 64;
constexpr uint32_t kSampleLengthMs = 2000;
constexpr uint32_t kAudioWindowSamples = (kSampleRateHz * kSampleLengthMs) / 1000; // 32000
constexpr uint32_t kWinLength = (kSampleRateHz * kWindowMs) / 1000; // 480
constexpr uint32_t kHopLength = (kSampleRateHz * kHopMs) / 1000; // 160
constexpr uint32_t kExpectedFrames = 1 + ((kAudioWindowSamples - kWinLength) / kHopLength); // 198
constexpr uint32_t kFftBins = kNfft / 2 + 1; // 257
constexpr uint32_t kMicDmaFrames = 512;

constexpr float kPcenAlpha = 0.65f;
constexpr float kPcenDelta = 0.10f;
constexpr float kPcenR = 0.30f;
constexpr float kPcenEps = 1.0e-6f;
constexpr float kPcenTimeConstantSec = 0.5f;

constexpr float kSilenceRmsThreshold = 0.002f;
constexpr float kPcenEnergyFloor = 0.01f;

constexpr float kBabyThreshold = 0.65f;
constexpr float kDogThreshold = 0.90f;
constexpr float kBabySuppressionFloor = 0.35f;
constexpr float kDogDominanceMargin = 0.15f;
constexpr float kLowConfidenceGate = 0.70f;
constexpr float kBabyDecay = 1.0f;
constexpr float kDogDecay = 0.3f;
constexpr float kBabyRequiredStreak = 1.0f;
constexpr float kDogRequiredStreak = 2.0f;

static int16_t mic_dma_buffer[2 * kMicDmaFrames];
static int16_t audio_ring[kAudioWindowSamples];
static volatile uint32_t audio_write_index = 0;
static bool mic_stream_started = false;
static float notch_prev_x = 0.0f;
static float notch_prev_y = 0.0f;

static float hann_window[kWinLength];
static int16_t mel_bin_points[kNMels + 2];
static float pcen_state[kNMels];
static int16_t feature_map_q12[kNMels][kExpectedFrames];
static float fft_input[kNfft];
static float fft_output[kNfft];
static float power_spectrum[kFftBins];
static arm_rfft_fast_instance_f32 fft_state;
static bool fft_ready = false;

static float baby_streak = 0.0f;
static float dog_streak = 0.0f;

static inline float hz_to_mel(float hz)
{
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel)
{
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void build_hann_window()
{
  for (uint32_t i = 0; i < kWinLength; ++i) {
    hann_window[i] = 0.5f - 0.5f * cosf((2.0f * kPi * i) / (kWinLength - 1));
  }
}

static void build_mel_basis()
{
  const float fmin = 50.0f;
  const float fmax = 8000.0f;
  const float mel_min = hz_to_mel(fmin);
  const float mel_max = hz_to_mel(fmax);

  for (uint32_t i = 0; i < (kNMels + 2); ++i) {
    const float mel = mel_min + (mel_max - mel_min) * (static_cast<float>(i) / (kNMels + 1));
    const float hz = mel_to_hz(mel);
    int32_t bin = static_cast<int32_t>(floorf(((kNfft + 1.0f) * hz) / kSampleRateHz));
    if (bin < 0) {
      bin = 0;
    }
    if (bin > static_cast<int32_t>(kFftBins - 1)) {
      bin = kFftBins - 1;
    }
    mel_bin_points[i] = static_cast<int16_t>(bin);
  }
}

static void reset_policy_and_pcen()
{
  memset(pcen_state, 0, sizeof(pcen_state));
  notch_prev_x = 0.0f;
  notch_prev_y = 0.0f;
  baby_streak = 0.0f;
  dog_streak = 0.0f;
}

static inline float get_ring_sample_norm(uint32_t ordered_index, uint32_t start_index)
{
  const uint32_t idx = (start_index + ordered_index) % kAudioWindowSamples;
  return static_cast<float>(audio_ring[idx]) / 32768.0f;
}

static void mic_buffer_ready_callback(const void *buffer, uint32_t n_frames)
{
  const int16_t *samples = static_cast<const int16_t *>(buffer);
  for (uint32_t i = 0; i < n_frames; ++i) {
    const float x = static_cast<float>(samples[i]) / 32768.0f;
    const float y = x - notch_prev_x + 0.95f * notch_prev_y;
    notch_prev_x = x;
    notch_prev_y = y;
    float yc = y;
    if (yc > 1.0f) {
      yc = 1.0f;
    } else if (yc < -1.0f) {
      yc = -1.0f;
    }
    audio_ring[audio_write_index] = static_cast<int16_t>(lrintf(yc * 32767.0f));
    audio_write_index = (audio_write_index + 1) % kAudioWindowSamples;
  }
}

static sl_status_t start_microphone_stream()
{
  sl_status_t st = sl_mic_init(kSampleRateHz, 1);
  if (st != SL_STATUS_OK) {
    return st;
  }
  st = sl_mic_start_streaming(
    mic_dma_buffer,
    kMicDmaFrames,
    static_cast<sl_mic_buffer_ready_callback_t>(mic_buffer_ready_callback));
  if (st != SL_STATUS_OK) {
    sl_mic_deinit();
    return st;
  }
  mic_stream_started = true;
  return SL_STATUS_OK;
}

static float compute_rms(uint32_t start_index)
{
  float sumsq = 0.0f;
  for (uint32_t i = 0; i < kAudioWindowSamples; ++i) {
    const float x = get_ring_sample_norm(i, start_index);
    sumsq += x * x;
  }
  return sqrtf(sumsq / kAudioWindowSamples);
}

static float compute_envelope_slope(uint32_t start_index)
{
  constexpr uint32_t frame_len = 400;  // 25 ms
  constexpr uint32_t half_sec = 8000;  // 0.5 s
  constexpr uint32_t env_frames = half_sec / frame_len;

  float env[env_frames];
  for (uint32_t i = 0; i < env_frames; ++i) {
    float frame_sumsq = 0.0f;
    const uint32_t offset = kAudioWindowSamples - half_sec + i * frame_len;
    for (uint32_t j = 0; j < frame_len; ++j) {
      const float x = get_ring_sample_norm(offset + j, start_index);
      frame_sumsq += x * x;
    }
    env[i] = sqrtf(frame_sumsq / frame_len);
  }

  float slope_sum = 0.0f;
  for (uint32_t i = 1; i < env_frames; ++i) {
    slope_sum += (env[i] - env[i - 1]);
  }
  return slope_sum / static_cast<float>(env_frames - 1);
}

static bool extract_silicon22_features(uint32_t start_index, float *out_mean, float *out_stddev)
{
  const float pcen_b = 1.0f - expf(-static_cast<float>(kHopLength) / (kPcenTimeConstantSec * kSampleRateHz));
  const float delta_r = powf(kPcenDelta, kPcenR);
  float sum = 0.0f;
  float sumsq = 0.0f;
  uint32_t count = 0;

  for (uint32_t t = 0; t < kExpectedFrames; ++t) {
    const uint32_t frame_start = t * kHopLength;
    for (uint32_t i = 0; i < kWinLength; ++i) {
      fft_input[i] = get_ring_sample_norm(frame_start + i, start_index) * hann_window[i];
    }
    for (uint32_t i = kWinLength; i < kNfft; ++i) {
      fft_input[i] = 0.0f;
    }

    arm_rfft_fast_f32(&fft_state, fft_input, fft_output, 0);
    power_spectrum[0] = fft_output[0] * fft_output[0];
    for (uint32_t k = 1; k < (kNfft / 2); ++k) {
      const float re = fft_output[2 * k];
      const float im = fft_output[2 * k + 1];
      power_spectrum[k] = re * re + im * im;
    }
    power_spectrum[kNfft / 2] = fft_output[1] * fft_output[1];

    for (uint32_t m = 0; m < kNMels; ++m) {
      float mel_energy = 0.0f;
      const int32_t left = mel_bin_points[m];
      const int32_t center = mel_bin_points[m + 1];
      const int32_t right = mel_bin_points[m + 2];
      if (center > left) {
        const float inv = 1.0f / static_cast<float>(center - left);
        for (int32_t k = left; k < center; ++k) {
          const float w = static_cast<float>(k - left) * inv;
          mel_energy += w * power_spectrum[k];
        }
      }
      if (right > center) {
        const float inv = 1.0f / static_cast<float>(right - center);
        for (int32_t k = center; k < right; ++k) {
          const float w = static_cast<float>(right - k) * inv;
          mel_energy += w * power_spectrum[k];
        }
      }
      if (mel_energy < 1.0e-10f) {
        mel_energy = 1.0e-10f;
      }

      const float m_prev = pcen_state[m];
      const float m_new = (1.0f - pcen_b) * m_prev + pcen_b * mel_energy;
      pcen_state[m] = m_new;

      const float denom = powf(kPcenEps + m_new, kPcenAlpha);
      const float pcen = powf((mel_energy / denom) + kPcenDelta, kPcenR) - delta_r;
      int32_t q = static_cast<int32_t>(lrintf(pcen * 4096.0f));
      if (q < -32768) {
        q = -32768;
      } else if (q > 32767) {
        q = 32767;
      }
      feature_map_q12[m][t] = static_cast<int16_t>(q);
      sum += pcen;
      sumsq += pcen * pcen;
      ++count;
    }
  }

  const float pcen_mean = sum / static_cast<float>(count);
  if (pcen_mean < kPcenEnergyFloor) {
    return false;
  }

  float variance = (sumsq / static_cast<float>(count)) - (pcen_mean * pcen_mean);
  if (variance < 1.0e-12f) {
    variance = 1.0e-12f;
  }
  const float stddev = sqrtf(variance);
  *out_mean = pcen_mean;
  *out_stddev = stddev;
  return true;
}

static sl_status_t fill_model_input(TfLiteTensor *input, float pcen_mean, float stddev)
{
  static bool shape_log_printed = false;

  if (input->dims->size != 4) {
    if (!shape_log_printed) {
      shape_log_printed = true;
      printf("Input tensor rank mismatch: got %d expected 4\r\n", input->dims->size);
    }
    return SL_STATUS_FAIL;
  }
  if (input->dims->data[0] != 1 || input->dims->data[1] != static_cast<int>(kNMels)
      || input->dims->data[2] != static_cast<int>(kExpectedFrames) || input->dims->data[3] != 1) {
    if (!shape_log_printed) {
      shape_log_printed = true;
      printf("Input tensor shape mismatch: got [%d,%d,%d,%d] expected [1,%lu,%lu,1], type=%d\r\n",
             input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
             static_cast<unsigned long>(kNMels),
             static_cast<unsigned long>(kExpectedFrames),
             static_cast<int>(input->type));
    }
    return SL_STATUS_FAIL;
  }

  if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int32_t zero_point = input->params.zero_point;
    if (scale <= 0.0f) {
      return SL_STATUS_FAIL;
    }

    int idx = 0;
    for (uint32_t m = 0; m < kNMels; ++m) {
      for (uint32_t t = 0; t < kExpectedFrames; ++t) {
        const float raw = static_cast<float>(feature_map_q12[m][t]) / 4096.0f;
        const float norm = (raw - pcen_mean) / (stddev + 1.0e-9f);
        int32_t q = static_cast<int32_t>(lrintf(norm / scale)) + zero_point;
        if (q < -128) {
          q = -128;
        } else if (q > 127) {
          q = 127;
        }
        input->data.int8[idx++] = static_cast<int8_t>(q);
      }
    }
    return SL_STATUS_OK;
  }

  if (input->type == kTfLiteFloat32) {
    int idx = 0;
    for (uint32_t m = 0; m < kNMels; ++m) {
      for (uint32_t t = 0; t < kExpectedFrames; ++t) {
        const float raw = static_cast<float>(feature_map_q12[m][t]) / 4096.0f;
        input->data.f[idx++] = (raw - pcen_mean) / (stddev + 1.0e-9f);
      }
    }
    return SL_STATUS_OK;
  }

  return SL_STATUS_FAIL;
}

static sl_status_t get_output_probabilities(const TfLiteTensor *output,
                                            float *baby_prob,
                                            float *dog_prob,
                                            float *unknown_prob)
{
  if (output->dims->size != 2 || output->dims->data[0] != 1 || output->dims->data[1] < 3) {
    return SL_STATUS_FAIL;
  }

  if (output->type == kTfLiteInt8) {
    const float scale = output->params.scale;
    const int32_t zero_point = output->params.zero_point;
    *baby_prob = (static_cast<float>(output->data.int8[0]) - zero_point) * scale;
    *dog_prob = (static_cast<float>(output->data.int8[1]) - zero_point) * scale;
    *unknown_prob = (static_cast<float>(output->data.int8[2]) - zero_point) * scale;
    return SL_STATUS_OK;
  }
  if (output->type == kTfLiteFloat32) {
    *baby_prob = output->data.f[0];
    *dog_prob = output->data.f[1];
    *unknown_prob = output->data.f[2];
    return SL_STATUS_OK;
  }
  return SL_STATUS_FAIL;
}

static inline void decay_streaks()
{
  baby_streak = fmaxf(0.0f, baby_streak - kBabyDecay);
  dog_streak = fmaxf(0.0f, dog_streak - kDogDecay);
}

static void apply_master_policy(float baby_prob, float dog_prob)
{
  if (fmaxf(baby_prob, dog_prob) < kLowConfidenceGate) {
    decay_streaks();
    return;
  }

  if (baby_prob >= kBabyThreshold) {
    baby_streak += 1.0f;
    dog_streak = fmaxf(0.0f, dog_streak - kDogDecay);
    return;
  }

  if ((dog_prob >= kDogThreshold)
      && ((dog_prob - baby_prob) > kDogDominanceMargin)
      && (baby_prob < kBabySuppressionFloor)) {
    dog_streak += 1.0f;
    baby_streak = fmaxf(0.0f, baby_streak - kBabyDecay);
    return;
  }

  decay_streaks();
}
} // namespace

static void audio_classifier_task(void *arg);

/***************************************************************************//**
 * Initialize audio classifier application.
 ******************************************************************************/
void audio_classifier_init(void)
{
  RTOS_ERR err;

  // Create Application Task
  char task_name[] = "audio classifier task";
  OSTaskCreate(&tcb,
               task_name,
               audio_classifier_task,
               DEF_NULL,
               TASK_PRIORITY,
               &stack[0],
               (TASK_STACK_SIZE / 10u),
               TASK_STACK_SIZE,
               0u,
               0u,
               DEF_NULL,
               (OS_OPT_TASK_STK_CLR),
               &err);

  EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));
}

/***************************************************************************//**
 * Audio classifier task function
 *
 * This function is executed by a Micrium OS task and does not return.
 *
 * @param arg ignored
 ******************************************************************************/
void audio_classifier_task(void *arg)
{
  RTOS_ERR err;
  (void)&arg;

  printf("Audio Classifier (silicon-22 compatible pipeline)\r\n");
  memset(audio_ring, 0, sizeof(audio_ring));
  reset_policy_and_pcen();
  build_hann_window();
  build_mel_basis();

  if (arm_rfft_fast_init_f32(&fft_state, kNfft) != ARM_MATH_SUCCESS) {
    printf("ERROR: FFT initialization failed\r\n");
    while (1) { ; }
  }
  fft_ready = true;

  if (start_microphone_stream() != SL_STATUS_OK) {
    printf("ERROR: microphone stream start failed\r\n");
    while (1) { ; }
  }

  const TfLiteTensor* input = sl_tflite_micro_get_input_tensor();
  const TfLiteTensor* output = sl_tflite_micro_get_output_tensor();
  if ((output->dims->size == 2) && (output->dims->data[0] == 1)) {
    category_count = output->dims->data[1];
  }
  if (category_count != category_label_count) {
    printf("WARNING: output classes=%d labels=%d\r\n", category_count, category_label_count);
  }

  if (!fft_ready) {
    printf("ERROR: FFT not ready\r\n");
    while (1) { ; }
  }

  if (!((input->type == kTfLiteInt8) || (input->type == kTfLiteFloat32))) {
    printf("ERROR: Input tensor must be int8 or float32\r\n");
    while (1) { ; }
  }
  if (!((output->type == kTfLiteInt8) || (output->type == kTfLiteFloat32))) {
    printf("ERROR: Output tensor must be int8 or float32\r\n");
    while (1) { ; }
  }

  // Add EM1 requirement to allow microphone sampling.
  sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);

  while (1) {
    // Delay task in order to do periodic inference.
    OSTimeDlyHMSM(0, 0, 0, INFERENCE_INTERVAL_MS, OS_OPT_TIME_PERIODIC, &err);
    EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));

    const uint32_t start_index = audio_write_index;
    const float rms = compute_rms(start_index);
    if (rms < kSilenceRmsThreshold) {
      memset(pcen_state, 0, sizeof(pcen_state));
      decay_streaks();
      sl_led_turn_off(&DETECTION_LED);
      sl_led_turn_off(&ACTIVITY_LED);
      continue;
    }

    const float slope = compute_envelope_slope(start_index);
    if (slope < 0.0005f) {
      dog_streak = fmaxf(0.0f, dog_streak - kDogDecay);
    }

    float pcen_mean = 0.0f;
    float pcen_stddev = 1.0f;
    if (!extract_silicon22_features(start_index, &pcen_mean, &pcen_stddev)) {
      decay_streaks();
      continue;
    }

    if (fill_model_input(sl_tflite_micro_get_input_tensor(), pcen_mean, pcen_stddev) != SL_STATUS_OK) {
      printf("ERROR: failed to fill model input tensor\r\n");
      continue;
    }

    if (sl_tflite_micro_get_interpreter()->Invoke() != kTfLiteOk) {
      printf("ERROR: model invoke failed\r\n");
      continue;
    }

    float baby_prob = 0.0f;
    float dog_prob = 0.0f;
    float unknown_prob = 0.0f;
    if (get_output_probabilities(sl_tflite_micro_get_output_tensor(), &baby_prob, &dog_prob, &unknown_prob) != SL_STATUS_OK) {
      printf("ERROR: failed to decode model output\r\n");
      continue;
    }

    apply_master_policy(baby_prob, dog_prob);

    const int32_t now_ms = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
    if (baby_streak >= kBabyRequiredStreak) {
      printf("[%ld ms] BABY_CRYING  baby=%.3f dog=%.3f unknown=%.3f streak=%.1f\r\n",
             now_ms, baby_prob, dog_prob, unknown_prob, baby_streak);
      app_ble_report_detection(0, static_cast<uint8_t>(fminf(fmaxf(baby_prob, 0.0f), 1.0f) * 100.0f));
      sl_led_turn_on(&DETECTION_LED);
      sl_led_turn_off(&ACTIVITY_LED);
    } else if (dog_streak >= kDogRequiredStreak) {
      printf("[%ld ms] DOG_BARKING  baby=%.3f dog=%.3f unknown=%.3f streak=%.1f\r\n",
             now_ms, baby_prob, dog_prob, unknown_prob, dog_streak);
      app_ble_report_detection(1, static_cast<uint8_t>(fminf(fmaxf(dog_prob, 0.0f), 1.0f) * 100.0f));
      sl_led_turn_on(&DETECTION_LED);
      sl_led_turn_off(&ACTIVITY_LED);
    } else if (baby_streak > 0.0f || dog_streak > 0.0f) {
      printf("[%ld ms] MAYBE       baby=%.3f dog=%.3f unknown=%.3f b_streak=%.1f d_streak=%.1f\r\n",
             now_ms, baby_prob, dog_prob, unknown_prob, baby_streak, dog_streak);
      sl_led_turn_off(&DETECTION_LED);
      sl_led_toggle(&ACTIVITY_LED);
    } else {
      sl_led_turn_off(&DETECTION_LED);
      sl_led_turn_off(&ACTIVITY_LED);
    }
  }
}

/***************************************************************************//**
 * Get the label for a certain category/class
 *
 * @param index
 *   index of the category/class
 *
 * @return
 *   pointer to the label string. The label is "?" if no corresponding label
 *   was found.
 ******************************************************************************/
const char * get_category_label(int index)
{
  if ((index >= 0) && (index < category_label_count)) {
    return category_labels[index];
  } else {
    return "?";
  }
}
