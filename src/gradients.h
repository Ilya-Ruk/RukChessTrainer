#ifndef GRADIENTS_H
#define GRADIENTS_H

//#include <limits.h>
#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad, int Epoch/*, float Min, float Max*/) {
  if (!grad->g) return;

  // AdaMax
/*
  grad->M = BETA1 * grad->M + (1.0f - BETA1) * grad->g;
  grad->V = fmaxf(BETA2 * grad->V, fabsf(grad->g));

  if (grad->V > 0.0f) {
    float delta = ALPHA / (1.0f - powf(BETA1, Epoch)) * (grad->M / grad->V);

    *v -= delta;
  }
*/
  // Adam

  grad->M = BETA1 * grad->M + (1.0f - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * grad->g * grad->g;

//  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  float M_Corrected = grad->M / (1.0f - powf(BETA1, Epoch));
  float V_Corrected = grad->V / (1.0f - powf(BETA2, Epoch));

  float delta = ALPHA * M_Corrected / (sqrtf(V_Corrected) + EPSILON);

  *v -= delta;
/*
  if (*v < Min) {
	*v = Min;
  }

  if (*v > Max) {
	*v = Max;
  }
*/
  grad->g = 0.0f;
}

void ApplyGradients(NN* nn, NNGradients* g, int Epoch) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeights[i], Epoch/*, (float)SHRT_MIN / (float)QUANTIZATION_PRECISION_IN, (float)SHRT_MAX / (float)QUANTIZATION_PRECISION_IN*/);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiases[i], Epoch/*, (float)SHRT_MIN / (float)QUANTIZATION_PRECISION_IN, (float)SHRT_MAX / (float)QUANTIZATION_PRECISION_IN*/);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeights[i], Epoch/*, (float)SHRT_MIN / (float)QUANTIZATION_PRECISION_OUT, (float)SHRT_MAX / (float)QUANTIZATION_PRECISION_OUT*/);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBias, Epoch/*, (float)SHRT_MIN / (float)QUANTIZATION_PRECISION_OUT, (float)SHRT_MAX / (float)QUANTIZATION_PRECISION_OUT*/);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif