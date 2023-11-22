#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad/*, int Epoch*/)
{
  if (grad->g == 0.0f) {
    return;
  }

  // Adam

  grad->M = BETA1 * grad->M + (1.0f - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * grad->g * grad->g;

  float delta = LR * grad->M / (sqrtf(grad->V) + EPSILON);

//  float M_Corrected = grad->M / (1.0f - powf(BETA1, Epoch));
//  float V_Corrected = grad->V / (1.0f - powf(BETA2, Epoch));

//  float delta = LR * M_Corrected / (sqrtf(V_Corrected) + EPSILON);

  *v -= delta;

  grad->g = 0.0f;
}

void ApplyGradients(NN* nn, NNGradients* g/*, int Epoch*/)
{
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) {
    UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeights[i]/*, Epoch*/);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiases[i]/*, Epoch*/);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeights[i]/*, Epoch*/);
  }

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBias/*, Epoch*/);
}

void ClearGradients(NNGradients* gradients)
{
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif