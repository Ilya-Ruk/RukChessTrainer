#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void UpdateAndApplyGradient(float* v, Gradient* grad, float g, int step, float min, float max)
{
  grad->M = BETA1 * grad->M + (1.0f - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * g * g;

  grad->Vmax = MAX(grad->Vmax, grad->V);

  float M_Corrected = grad->M / (1.0f - powf(BETA1, step));
  float V_Corrected = grad->Vmax / (1.0f - powf(BETA2, step));

  *v -= alpha * M_Corrected / (sqrtf(V_Corrected) + EPSILON);

  *v = MIN(MAX(min, *v), max);
}

void ApplyGradients(NN* nn, NNGradients* gradients, BatchGradients* local, uint8_t* active, int step)
{
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT; i++) {
    if (!active[i]) {
      continue;
    }

    for (int j = 0; j < N_HIDDEN; j++) {
      int idx = i * N_HIDDEN + j;

      float g = 0.0f;

      for (int t = 0; t < THREADS; t++) {
        g += local[t].inputWeights[idx];
      }

      UpdateAndApplyGradient(&nn->inputWeights[idx], &gradients->inputWeights[idx], g, step, -3.89f, 3.89f);
    }
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].inputBiases[i];
    }

    UpdateAndApplyGradient(&nn->inputBiases[i], &gradients->inputBiases[i], g, step, -3.89f, 3.89f);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    float g = 0.0f;

    for (int t = 0; t < THREADS; t++) {
      g += local[t].outputWeights[i];
    }

    UpdateAndApplyGradient(&nn->outputWeights[i], &gradients->outputWeights[i], g, step, -2.0f, 2.0f);
  }

  float g = 0.0f;

  for (int t = 0; t < THREADS; t++) {
    g += local[t].outputBias;
  }

  UpdateAndApplyGradient(&nn->outputBias, &gradients->outputBias, g, step, -2.0f, 2.0f);
}

void ClearGradients(NNGradients* gradients)
{
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif