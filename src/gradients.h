#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

static void UpdateAndApplyGradient(float* v, Gradient* grad, float g)
{
  if (g == 0.0f) {
    return;
  }

  grad->M = BETA1 * grad->M + (1.0f - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * g * g;

  *v -= ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);
}

void ApplyGradients(NN* nn, NNGradients* gradients, BatchGradients* local)
{
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      int idx = i * N_HIDDEN + j;

      float g = 0.0f;

      for (int t = 0; t < THREADS; t++) {
        g += local[t].inputWeights[idx];
      }

      UpdateAndApplyGradient(&nn->inputWeights[idx], &gradients->inputWeights[idx], g);
    }
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].inputBiases[i];
    }

    UpdateAndApplyGradient(&nn->inputBiases[i], &gradients->inputBiases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].outputWeights[i];
    }

    UpdateAndApplyGradient(&nn->outputWeights[i], &gradients->outputWeights[i], g);
  }

  float g = 0.0f;

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) {
    g += local[t].outputBias;
  }

  UpdateAndApplyGradient(&nn->outputBias, &gradients->outputBias, g);
}

void ClearGradients(NNGradients* gradients)
{
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif