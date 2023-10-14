#include "trainer.h"

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "bits.h"
#include "board.h"
#include "data.h"
#include "gradients.h"
#include "nn.h"
#include "random.h"
#include "util.h"

static float TotalError(DataSet* data, NN* nn)
{
  float e = 0.0f;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
  for (int i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    Features f[1];
    NNAccumulators activations[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, activations, 0);

    e += Error(Sigmoid(activations->output), board);
  }

  return e / data->n;
}

static void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local)
{
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) {
    memset(&local[t], 0, sizeof(BatchGradients));
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    Features f[1];
    NNAccumulators activations[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, activations, 1);

    float out = Sigmoid(activations->output);

    // Loss calculations

    float outputLoss = SigmoidPrime(out) * ErrorPrime(out, &board);

    float hiddenLosses2[N_HIDDEN_2];

    for (int i = 0; i < N_HIDDEN_2; i++) {
      hiddenLosses2[i] = outputLoss * nn->outputWeights[i] * CReLUPrime(activations->acc2[i]);
    }

    float hiddenLosses1[2][N_HIDDEN_1] = {0};

    for (int i = 0; i < N_HIDDEN_1; i++) {
      for (int j = 0; j < N_HIDDEN_2; j++) {
        hiddenLosses1[board.stm][i] += hiddenLosses2[j] * nn->hiddenWeights[j * 2 * N_HIDDEN_1 + i] * CReLUPrime(activations->acc1[board.stm][i]);
        hiddenLosses1[board.stm ^ 1][i] += hiddenLosses2[j] * nn->hiddenWeights[j * 2 * N_HIDDEN_1 + i + N_HIDDEN_1] * CReLUPrime(activations->acc1[board.stm ^ 1][i]);
      }
    }

    // Output layer gradients

    local[t].outputBias += outputLoss;

    for (int i = 0; i < N_HIDDEN_2; i++) {
      local[t].outputWeights[i] += activations->acc2[i] * outputLoss;
    }

    // Hidden layer gradients

    for (int i = 0; i < N_HIDDEN_2; i++) {
      local[t].hiddenBiases[i] += hiddenLosses2[i];
    }

    for (int i = 0; i < N_HIDDEN_1; i++) {
      for (int j = 0; j < N_HIDDEN_2; j++) {
        local[t].hiddenWeights[j * 2 * N_HIDDEN_1 + i] += activations->acc1[board.stm][i] * hiddenLosses2[j];
        local[t].hiddenWeights[j * 2 * N_HIDDEN_1 + i + N_HIDDEN_1] += activations->acc1[board.stm ^ 1][i] * hiddenLosses2[j];
      }
    }

    // Input layer gradients

    for (int i = 0; i < N_HIDDEN_1; i++) {
      local[t].inputBiases[i] += hiddenLosses1[board.stm][i] + hiddenLosses1[board.stm ^ 1][i];
    }

    for (int i = 0; i < f->n; i++) {
      for (int j = 0; j < N_HIDDEN_1; j++) {
        local[t].inputWeights[f->features[board.stm][i] * N_HIDDEN_1 + j] += hiddenLosses1[board.stm][j];
        local[t].inputWeights[f->features[board.stm ^ 1][i] * N_HIDDEN_1 + j] += hiddenLosses1[board.stm ^ 1][j];
      }
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN_1; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->inputWeights[i].g += local[t].inputWeights[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_1; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->inputBiases[i].g += local[t].inputBiases[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_1 * N_HIDDEN_2 * 2; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->hiddenWeights[i].g += local[t].hiddenWeights[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->hiddenBiases[i].g += local[t].hiddenBiases[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->outputWeights[i].g += local[t].outputWeights[i];
    }
  }

  for (int t = 0; t < THREADS; t++) {
    g->outputBias.g += local[t].outputBias;
  }
}

int main(int argc, char** argv)
{
  SeedRandom();

  // Read command options

  int c;
  int m = 0;

  char nnPath[128] = {0};

  char validPath[128] = {0};
  char trainPath[128] = {0};

  while ((c = getopt(argc, argv, "n:v:t:m")) != -1) {
    switch (c) {
      case 'n':
        strcpy(nnPath, optarg);
        break;

      case 'v':
        strcpy(validPath, optarg);
        break;

      case 't':
        strcpy(trainPath, optarg);
        break;

      case 'm':
        m = 1;
        break;

      default:
        return 1;
    }
  }

  if (!validPath[0]) {
    printf("No valid data file specified!\n");

    return 1;
  }

  if (!trainPath[0]) {
    printf("No train data file specified!\n");

    return 1;
  }

  // Load or generate net

  NN* nn;

  if (!nnPath[0]) {
    printf("No net specified, generating a random net...\n");

    nn = LoadRandomNN();

    printf("No net specified, generating a random net...DONE\n\n");
  } else {
    printf("Loading net from %s...\n", nnPath);

    nn = LoadNN(nnPath);

    printf("Loading net from %s...DONE\n\n", nnPath);
  }

  // Load valid data

  printf("Loading valid data from %s...\n", validPath);

  DataSet* validData = malloc(sizeof(DataSet));

  validData->n = 0;
  validData->entries = malloc(sizeof(Board) * MAX_VALID_POSITIONS);

  LoadEntries(validPath, validData, MAX_VALID_POSITIONS);

  printf("Loading valid data from %s...DONE\n\n", validPath);

  // Accumulator max.
  // Output min. and max.

  if (m) {
    float maxAcc = -FLT_MAX;

    float minOut = FLT_MAX;
    float maxOut = -FLT_MAX;

    for (int i = 0; i < validData->n; i++) {
      Board board = validData->entries[i];

      Features f[1];
      NNAccumulators activations[1];

      ToFeatures(&board, f);
      NNPredict(nn, f, board.stm, activations, 0);

      for (int j = 0; j < 2; j++) { // WHITE, BLACK
        for (int k = 0; k < N_HIDDEN_1; k++) {
          float acc = activations->acc1[j][k];

          if (acc > maxAcc) {
            maxAcc = acc;
          }
        }
      }

      float out = activations->output;

      if (out < minOut) {
        minOut = out;
      }

      if (out > maxOut) {
        maxOut = out;
      }
    }

    printf("maxAcc = %.8f\n", maxAcc);
    printf("minOut = %.8f maxOut = %.8f\n", minOut, maxOut);

    return 0;
  }

  // Load train data

  printf("Loading train data from %s...\n", trainPath);

  DataSet* trainData = malloc(sizeof(DataSet));

  trainData->n = 0;
  trainData->entries = malloc(sizeof(Board) * MAX_TRAIN_POSITIONS);

  LoadEntries(trainPath, trainData, MAX_TRAIN_POSITIONS);

  printf("Loading train data from %s...DONE\n\n", trainPath);

  // Prepare gradients

  NNGradients* gradients = malloc(sizeof(NNGradients));

  ClearGradients(gradients);

  BatchGradients* local = malloc(sizeof(BatchGradients) * THREADS);

  // Calculate valid error

  printf("Calculating valid error...\n");

  float validError = TotalError(validData, nn);

  printf("Calculating valid error...DONE\n\n");

  printf("Valid error: %.8f\n\n", validError);

  // Train net

  for (int epoch = 1; epoch <= 250; epoch++) {
    long epochStartTime = GetTimeMS();

    // Shuffle train data

    printf("Shuffling train data...\n");

    ShuffleData(trainData);

    printf("Shuffling train data...DONE\n\n");

    // Train net

    printf("Train net...\n");

    int batches = trainData->n / BATCH_SIZE;

    for (int b = 0; b < batches; b++) {
      Train(b, trainData, nn, gradients, local);

      ApplyGradients(nn, gradients/*, epoch*/);

      if (((b + 1) % 1000) == 0) {
        printf("Batch: %5d / %5d\n", b + 1, batches);
      }
    }

    printf("Train net...DONE\n\n");

    // Save net

    printf("Save net...\n");

    char buffer[64];

    sprintf(buffer, "../Nets/rukchess_%03d.nnue", epoch);

    SaveNN(nn, buffer);

    printf("Save net...DONE\n\n");

    // Calculate valid error

    printf("Calculating valid error...\n");

    float newValidError = TotalError(validData, nn);

    printf("Calculating valid error...DONE\n\n");

    // Print epoch, valid error and delta, time and speed

    long epochEndTime = GetTimeMS();

    printf("Epoch: %3d Valid error: %.8f (%+.8f) Time: %ld sec Speed: %7.0f pos/sec\n\n",
           epoch, newValidError, newValidError - validError,
           (epochEndTime - epochStartTime) / 1000,
           1000.0f * trainData->n / (epochEndTime - epochStartTime));

    // Update valid error

    validError = newValidError;
  }

  return 0;
}