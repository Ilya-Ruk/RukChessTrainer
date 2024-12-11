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

#pragma omp parallel for schedule(static) num_threads(THREADS) reduction(+ : e)
  for (int i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    Features f[1];
    NNAccumulators activations[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, activations);

    e += Error(Sigmoid(activations->output), board);
  }

  return e / data->n;
}

static void Train(int batch, DataSet* data, NN* nn, BatchGradients* local)
{
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) {
    memset(&local[t], 0, sizeof(BatchGradients));
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board* board = &data->entries[n + batch * BATCH_SIZE];

    Features f[1];
    NNAccumulators activations[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, activations);

    float out = Sigmoid(activations->output);

    // Loss calculations

    float outputLoss = SigmoidPrime(out) * ErrorPrime(out, board);

    float hiddenLosses[2][N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++) {
      hiddenLosses[board->stm][i] = outputLoss * nn->outputWeights[i] * ReLUPrime(activations->acc[board->stm][i]);
      hiddenLosses[board->stm ^ 1][i] = outputLoss * nn->outputWeights[i + N_HIDDEN] * ReLUPrime(activations->acc[board->stm ^ 1][i]);
    }

    // Output layer gradients

    local[t].outputBias += outputLoss;

    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].outputWeights[i] += activations->acc[board->stm][i] * outputLoss;
      local[t].outputWeights[i + N_HIDDEN] += activations->acc[board->stm ^ 1][i] * outputLoss;
    }

    // Input layer gradients

    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].inputBiases[i] += hiddenLosses[board->stm][i] + hiddenLosses[board->stm ^ 1][i];
    }

    for (int i = 0; i < f->n; i++) {
      int f1 = f->features[board->stm][i];
      int f2 = f->features[board->stm ^ 1][i];

      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].inputWeights[f1 * N_HIDDEN + j] += hiddenLosses[board->stm][j];
        local[t].inputWeights[f2 * N_HIDDEN + j] += hiddenLosses[board->stm ^ 1][j];
      }
    }
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
      Board* board = &validData->entries[i];

      Features f[1];
      NNAccumulators activations[1];

      ToFeatures(board, f);
      NNPredict(nn, f, board->stm, activations);

      for (int j = 0; j < N_HIDDEN; j++) {
        float acc0 = activations->acc[WHITE][j];
        float acc1 = activations->acc[BLACK][j];

        if (acc0 > maxAcc) {
          maxAcc = acc0;
        }

        if (acc1 > maxAcc) {
          maxAcc = acc1;
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

  // Calculate train and valid errors

  printf("Calculate train and valid errors...\n");

  float trainError = TotalError(trainData, nn);
  float validError = TotalError(validData, nn);

  printf("Calculate train and valid errors...DONE\n\n");

  // Print train and valid errors

  printf("Train error: %.8f Valid error: %.8f\n\n", trainError, validError);

  // Train net

  FILE* fp = fopen("../Nets/train_error_log.txt", "w");

  if (fp == NULL) {
    printf("Unable to create file: train_error_log.txt!\n");

    exit(1);
  }

  for (int epoch = 1; epoch <= MAX_EPOCH; epoch++) {
    long epochStartTime = GetTimeMS();

    // Shuffle train data

    printf("Shuffling train data...\n");

    ShuffleData(trainData);

    printf("Shuffling train data...DONE\n\n");

    // Train net

    printf("Train net...\n");

    int batches = trainData->n / BATCH_SIZE;

    for (int b = 0; b < batches; b++) {
      Train(b, trainData, nn, local);

      ApplyGradients(nn, gradients, local, epoch);

      if ((b + 1) % 1000 == 0) {
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

    // Calculate train and valid errors

    printf("Calculate train and valid errors...\n");

    float newTrainError = TotalError(trainData, nn);
    float newValidError = TotalError(validData, nn);

    printf("Calculate train and valid errors...DONE\n\n");

    // Print epoch, train and valid errors with delta, time and speed

    long epochEndTime = GetTimeMS();

    printf("Epoch: %3d Train error: %.8f (%+.8f) Valid error: %.8f (%+.8f) Time: %ld sec Speed: %7.0f pos/sec\n\n",
           epoch,
           newTrainError, newTrainError - trainError,
           newValidError, newValidError - validError,
           (epochEndTime - epochStartTime) / 1000,
           1000.0f * trainData->n / (epochEndTime - epochStartTime));

    // Save epoch, train and valid errors

    fprintf(fp, "%d;%.8f;%.8f\n", epoch, newTrainError, newValidError);

    fflush(fp);

    // Update train and valid errors

    trainError = newTrainError;
    validError = newValidError;
  }

  fclose(fp);

  return 0;
}