#ifndef TRAINER_H
#define TRAINER_H

#include "data.h"
#include "nn.h"

#define ERR_THREADS 30
#define THREADS 16
#define BATCH_SIZE 16384

#define ALPHA 0.01f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient featureWeightGradients[N_FEATURES * N_HIDDEN];
  Gradient hiddenWeightGradients[N_HIDDEN * N_OUTPUT];
  Gradient hiddenBiasGradients[N_HIDDEN];
  Gradient outputBiasGradients[N_OUTPUT];
} NNGradients;

typedef struct {
  float featureWeightGradients[N_FEATURES * N_HIDDEN];
  float hiddenWeightGradients[N_HIDDEN * N_OUTPUT];
  float hiddenBiasGradients[N_HIDDEN];
  float outputBiasGradients[N_OUTPUT];
} BatchGradients;

typedef struct {
  int start, n;
  NNActivations activations;
  DataSet* data;
  NN* nn;
  NNGradients* gradients;
} UpdateGradientsJob;

typedef struct {
  int start, n;
  float error;
  DataSet* data;
  NN* nn;
} CalculateErrorJob;

float Error(float result, DataEntry* entry);
float ErrorGradient(float result, DataEntry* entry);
float TotalError(DataSet* data, NN* nn);
void* CalculateError(void* arg);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, NNGradients* threadGradients);
void* CalculateGradients(void* arg);
void UpdateNetwork(NN* nn, NNGradients* g);
void UpdateAndApplyGradient(float* v, Gradient* grad);
void ClearGradients(NNGradients* gradients);

#endif