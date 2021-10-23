#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

INLINE void UpdateAndApplyGradient(float* v, Gradient* grad) {
  if (!grad->g)
    return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;

  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;

  grad->g = 0;
}

INLINE void ApplyGradients(NN* nn, NNGradients* g) {
#pragma omp parallel for schedule(static, N_FEATURES* N_HIDDEN / THREADS) num_threads(THREADS)
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->featureWeights[i], &g->featureWeightGradients[i]);

#pragma omp parallel for schedule(static, N_HIDDEN / THREADS) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->hiddenBiases[i], &g->hiddenBiasGradients[i]);

#pragma omp parallel for schedule(static, 2 * N_HIDDEN / THREADS) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++)
    UpdateAndApplyGradient(&nn->hiddenWeights[i], &g->hiddenWeightGradients[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBiasGradient);
}

INLINE void ClearGradients(NNGradients* gradients) {
  memset(gradients->featureWeightGradients, 0, sizeof(gradients->featureWeightGradients));
  memset(gradients->hiddenBiasGradients, 0, sizeof(gradients->hiddenBiasGradients));
  memset(gradients->hiddenWeightGradients, 0, sizeof(gradients->hiddenWeightGradients));
  gradients->outputBiasGradient = (Gradient){.g = 0.0f, .M = 0.0f, .V = 0.0f};
}

#endif