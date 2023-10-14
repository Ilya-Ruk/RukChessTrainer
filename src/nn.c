#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "random.h"
#include "types.h"
#include "util.h"

const int NETWORK_MAGIC = 'B' | 'R' << 8 | 'K' << 16 | 'R' << 24;

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results, int train)
{
  // Input biases

  memcpy(results->acc1[WHITE], nn->inputBiases, sizeof(float) * N_HIDDEN_1);
  memcpy(results->acc1[BLACK], nn->inputBiases, sizeof(float) * N_HIDDEN_1);

  // Input weights

  for (int i = 0; i < f->n; i++) {
    for (int j = 0; j < N_HIDDEN_1; j++) {
      results->acc1[WHITE][j] += nn->inputWeights[f->features[WHITE][i] * N_HIDDEN_1 + j];
      results->acc1[BLACK][j] += nn->inputWeights[f->features[BLACK][i] * N_HIDDEN_1 + j];
    }
  }

  // ReLU

  CReLU(results->acc1[WHITE], NUM_REGS_1);
  CReLU(results->acc1[BLACK], NUM_REGS_1);

  // Dropout

#ifdef DROPOUT
  if (train) {
    for (int i = 0; i < N_HIDDEN; i++) {
      float rnd1 = (float)rand() / RAND_MAX;

      if (rnd1 > DROPOUT_P) {
        results->acc1[WHITE][i] /= DROPOUT_Q;
      }
      else {
        results->acc1[WHITE][i] = 0.0f;
      }

      float rnd2 = (float)rand() / RAND_MAX;

      if (rnd2 > DROPOUT_P) {
        results->acc1[BLACK][i] /= DROPOUT_Q;
      }
      else {
        results->acc1[BLACK][i] = 0.0f;
      }
    }
  }
#else
  (void)train;
#endif // DROPOUT

  // Hidden biases

  memcpy(results->acc2, nn->hiddenBiases, sizeof(float) * N_HIDDEN_2);

  // Hidden weights

  for (int i = 0; i < N_HIDDEN_2; i++) {
    results->acc2[i] += DotProduct(results->acc1[stm], &nn->hiddenWeights[i * 2 * N_HIDDEN_1], NUM_REGS_1) +
                        DotProduct(results->acc1[stm ^ 1], &nn->hiddenWeights[i * 2 * N_HIDDEN_1 + N_HIDDEN_1], NUM_REGS_1);
  }

  // ReLU

  CReLU(results->acc2, NUM_REGS_2);

  // Output

  results->output = DotProduct(results->acc2, nn->outputWeights, NUM_REGS_2) +
                    nn->outputBias;
}

NN* LoadNN(char* path)
{
  FILE* fp = fopen(path, "rb");

  if (fp == NULL) {
    printf("Unable to read network at %s!\n", path);

    exit(1);
  }

  int magic;
  fread(&magic, 4, 1, fp);

  if (magic != NETWORK_MAGIC) {
    printf("Magic header does not match!\n");

    exit(1);
  }

  uint64_t hash;
  fread(&hash, sizeof(uint64_t), 1, fp);

  printf("Reading network with hash %lx\n", hash);

  NN* nn = AlignedMalloc(sizeof(NN));

  fread(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN_1, fp);
  fread(nn->inputBiases, sizeof(float), N_HIDDEN_1, fp);
  fread(nn->hiddenWeights, sizeof(float), N_HIDDEN_1 * N_HIDDEN_2 * 2, fp);
  fread(nn->hiddenBiases, sizeof(float), N_HIDDEN_2, fp);
  fread(nn->outputWeights, sizeof(float), N_HIDDEN_2, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN(void)
{
  srand(time(NULL));

  NN* nn = AlignedMalloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN_1; i++) {
    nn->inputWeights[i] = RandomGaussian(0.0f, sqrtf(1.0f / 32));
  }

  for (int i = 0; i < N_HIDDEN_1; i++) {
    nn->inputBiases[i] = 0.0f;
  }

  for (int i = 0; i < N_HIDDEN_1 * N_HIDDEN_2 * 2; i++) {
    nn->hiddenWeights[i] = RandomGaussian(0.0f, sqrtf(1.0f / N_HIDDEN_1));
  }

  for (int i = 0; i < N_HIDDEN_2; i++) {
    nn->hiddenBiases[i] = 0.0f;
  }

  for (int i = 0; i < N_HIDDEN_2; i++) {
    nn->outputWeights[i] = RandomGaussian(0.0f, sqrtf(1.0f / N_HIDDEN_2));
  }

  nn->outputBias = 0.0f;

  return nn;
}

void SaveNN(NN* nn, char* path)
{
  FILE* fp = fopen(path, "wb");

  if (fp == NULL) {
    printf("Unable to save network to %s!\n", path);

    exit(1);
  }

  fwrite(&NETWORK_MAGIC, sizeof(int), 1, fp);

  uint64_t hash = NetworkHash(nn);
  fwrite(&hash, sizeof(uint64_t), 1, fp);

  fwrite(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN_1, fp);
  fwrite(nn->inputBiases, sizeof(float), N_HIDDEN_1, fp);
  fwrite(nn->hiddenWeights, sizeof(float), N_HIDDEN_1 * N_HIDDEN_2 * 2, fp);
  fwrite(nn->hiddenBiases, sizeof(float), N_HIDDEN_2, fp);
  fwrite(nn->outputWeights, sizeof(float), N_HIDDEN_2, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}