#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

static void UpdateAndApplyGradientWithEpoch(float* v, Gradient* grad, float g, int epoch)
{
  grad->M = BETA1 * grad->M + (1.0f - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * g * g;

  float M_Corrected = grad->M / (1.0f - powf(BETA1, epoch));
  float V_Corrected = grad->V / (1.0f - powf(BETA2, epoch));

  *v -= ALPHA * M_Corrected / (sqrtf(V_Corrected) + EPSILON);
}

void ApplyGradients(NN* nn, NNGradients* gradients, BatchGradients* local, int epoch)
{
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      int idx = i * N_HIDDEN + j;

      float g = 0.0f;

      for (int t = 0; t < THREADS; t++) {
        g += local[t].inputWeights[idx];
      }

      UpdateAndApplyGradientWithEpoch(&nn->inputWeights[idx], &gradients->inputWeights[idx], g, epoch);
    }
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].inputBiases[i];
    }

    UpdateAndApplyGradientWithEpoch(&nn->inputBiases[i], &gradients->inputBiases[i], g, epoch);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].outputWeights[i];
    }

    UpdateAndApplyGradientWithEpoch(&nn->outputWeights[i], &gradients->outputWeights[i], g, epoch);
  }

  float g = 0.0f;

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) {
    g += local[t].outputBias;
  }

  UpdateAndApplyGradientWithEpoch(&nn->outputBias, &gradients->outputBias, g, epoch);
}

void ClearGradients(NNGradients* gradients)
{
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif