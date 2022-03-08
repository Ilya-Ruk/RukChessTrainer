#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local);

INLINE float Error(float r, Board* b) { return powf(fabsf(r - (float)b->wdl / 2.0f), 2.5f); }

INLINE float ErrorGradient(float r, Board* b) { return 2.5f * (r - (float)b->wdl / 2.0f) * sqrtf(fabsf(r - (float)b->wdl / 2.0f)); }

//INLINE float Error(float r, Board* b) { return powf(r - (float)b->wdl / 2.0f, 2.0f); }

//INLINE float ErrorGradient(float r, Board* b) { return 2.0f * (r - (float)b->wdl / 2.0f); }

#endif