#ifndef NN_H
#define NN_H

#include <immintrin.h>

#include "types.h"
#include "util.h"

#define H(h, v) ((h) + (324723947ULL + (v))) ^ 93485734985ULL

INLINE uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;

  for (int i = 0; i < N_HIDDEN * N_INPUT; i++) {
    hash = H(hash, (int)nn->inputWeights[i]);
  }

  for (int i = 0; i < N_HIDDEN; i++) {
    hash = H(hash, (int)nn->inputBiases[i]);
  }

  for (int i = 0; i < N_HIDDEN * 2; i++) {
    hash = H(hash, (int)nn->outputWeights[i]);
  }

  hash = H(hash, (int)nn->outputBias);

  return hash;
}

INLINE void ReLU(float* v, const int n) {
  const int width = sizeof(__m256) / sizeof(float);
  const int chunks = n / width;

  const __m256 zero = _mm256_setzero_ps();

  __m256* vector = (__m256*)v;

  for (int j = 0; j < chunks; j++) {
    vector[j] = _mm256_max_ps(zero, vector[j]);
  }
}

INLINE float ReLUPrime(float s) {
  return s > 0.0f;
}
/*
INLINE void CReLU(float* v, const int n) {
  const int width = sizeof(__m256) / sizeof(float);
  const int chunks = n / width;

  const __m256 zero = _mm256_setzero_ps();
  const __m256 max = _mm256_set1_ps(CRELU_MAX);

  __m256* vector = (__m256*)v;

  for (int j = 0; j < chunks; j++) {
    vector[j] = _mm256_min_ps(max, _mm256_max_ps(zero, vector[j]));
  }
}

INLINE float CReLUPrime(float s) {
  return s > 0.0f && s < CRELU_MAX;
}
*/
INLINE float DotProduct(float* v1, float* v2, const int n) {
  const int width = sizeof(__m256) / sizeof(float);
  const int chunks = n / width;

  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  __m256* vector1 = (__m256*)v1;
  __m256* vector2 = (__m256*)v2;

  for (int j = 0; j < chunks; j += 2) {
    s0 = _mm256_add_ps(_mm256_mul_ps(vector1[j], vector2[j]), s0);
    s1 = _mm256_add_ps(_mm256_mul_ps(vector1[j + 1], vector2[j + 1]), s1);
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));

  return _mm_cvtss_f32(r1);
}

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results);

NN* LoadNN(char* path);
NN* LoadRandomNN(void);

void SaveNN(NN* nn, char* path);

#endif