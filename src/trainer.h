#ifndef TRAINER_H
#define TRAINER_H

#include <math.h>

#include "types.h"
#include "util.h"

INLINE float Error(float r, Board* b) {
  return powf(fabsf(r - (float)b->wdl / 2.0f), 2.5f);
}

INLINE float ErrorPrime(float r, Board* b) {
  return 2.5f * (r - (float)b->wdl / 2.0f) * sqrtf(fabsf(r - (float)b->wdl / 2.0f));
}

INLINE float Sigmoid(float s) {
  return 1.0f / (1.0f + expf(-s * SS));
}

INLINE float SigmoidPrime(float s) {
  return s * (1.0f - s) * SS;
}

#endif