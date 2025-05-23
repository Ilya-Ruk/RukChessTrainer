#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "random.h"
#include "types.h"
#include "util.h"

const int NETWORK_MAGIC = 'B' | 'R' << 8 | 'K' << 16 | 'R' << 24;

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results)
{
  // Input biases

  memcpy(results->acc[WHITE], nn->inputBiases, sizeof(float) * N_HIDDEN);
  memcpy(results->acc[BLACK], nn->inputBiases, sizeof(float) * N_HIDDEN);

  // Input weights

  for (int i = 0; i < f->n; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      results->acc[WHITE][j] += nn->inputWeights[f->features[WHITE][i] * N_HIDDEN + j];
      results->acc[BLACK][j] += nn->inputWeights[f->features[BLACK][i] * N_HIDDEN + j];
    }
  }

  // ReLU

  ReLU(results->acc[WHITE]);
  ReLU(results->acc[BLACK]);

  // Output

  results->output = DotProduct(results->acc[stm], nn->outputWeights) +
                    DotProduct(results->acc[stm ^ 1], nn->outputWeights + N_HIDDEN) +
                    nn->outputBias;
}

NN* LoadNN(char* path)
{
  FILE* fp = fopen(path, "rb");

  if (fp == NULL) {
    printf("Unable to read network: %s!\n", path);

    exit(1);
  }

  int magic;
  fread(&magic, 4, 1, fp);

  if (magic != NETWORK_MAGIC) {
    printf("Magic header does not match!\n");

    fclose(fp);

    exit(1);
  }

  uint64_t hash;
  fread(&hash, sizeof(uint64_t), 1, fp);

  printf("Reading network with hash %llx\n", hash);

  NN* nn = AlignedMalloc(sizeof(NN));

  fread(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN, fp);
  fread(nn->inputBiases, sizeof(float), N_HIDDEN, fp);
  fread(nn->outputWeights, sizeof(float), N_HIDDEN * 2, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN(void)
{
  srand(time(NULL));

  NN* nn = AlignedMalloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) {
    nn->inputWeights[i] = RandomGaussian(0.0f, sqrtf(1.0f / 32));
  }

  for (int i = 0; i < N_HIDDEN; i++) {
    nn->inputBiases[i] = 0.0f;
  }

  for (int i = 0; i < N_HIDDEN * 2; i++) {
    nn->outputWeights[i] = RandomGaussian(0.0f, sqrtf(1.0f / N_HIDDEN));
  }

  nn->outputBias = 0.0f;

  return nn;
}

void SaveNN(NN* nn, char* path)
{
  FILE* fp = fopen(path, "wb");

  if (fp == NULL) {
    printf("Unable to save network: %s!\n", path);

    exit(1);
  }

  fwrite(&NETWORK_MAGIC, sizeof(int), 1, fp);

  uint64_t hash = NetworkHash(nn);
  fwrite(&hash, sizeof(uint64_t), 1, fp);

  fwrite(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN, fp);
  fwrite(nn->inputBiases, sizeof(float), N_HIDDEN, fp);
  fwrite(nn->outputWeights, sizeof(float), N_HIDDEN * 2, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}