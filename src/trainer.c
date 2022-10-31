#include "trainer.h"

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

static float TotalError(DataSet* data, NN* nn) {
  float e = 0.0f;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
  for (int i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    NNAccumulators results[1];
    Features f[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, results);

    e += Error(Sigmoid(results->output), board);
  }

  return e / data->n;
}

static void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) {
    memset(&local[t], 0, sizeof(BatchGradients));
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    NNAccumulators activations[1];
    Features f[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, activations);

    float out = Sigmoid(activations->output);

    // Loss calculations

    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &board);

    float hiddenLosses[2][N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++) {
      hiddenLosses[board.stm][i] = outputLoss * nn->outputWeights[i] * ReLUPrime(activations->acc[board.stm][i]);
      hiddenLosses[board.stm ^ 1][i] = outputLoss * nn->outputWeights[i + N_HIDDEN] * ReLUPrime(activations->acc[board.stm ^ 1][i]);
    }

    // Output layer gradients

    local[t].outputBias += outputLoss;

    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].outputWeights[i] += activations->acc[board.stm][i] * outputLoss;
      local[t].outputWeights[i + N_HIDDEN] += activations->acc[board.stm ^ 1][i] * outputLoss;
    }

    // Input layer gradients

    for (int i = 0; i < N_HIDDEN; i++) {
//      float stmLasso = LAMBDA * (activations->acc[board.stm][i] > 0.0f);
//      float xstmLasso = LAMBDA * (activations->acc[board.stm ^ 1][i] > 0.0f);

      local[t].inputBiases[i] += hiddenLosses[board.stm][i] + hiddenLosses[board.stm ^ 1][i]/* + stmLasso + xstmLasso*/;
    }

    for (int i = 0; i < f->n; i++) {
      for (int j = 0; j < N_HIDDEN; j++) {
//        float stmLasso = LAMBDA * (activations->acc[board.stm][j] > 0.0f);
//        float xstmLasso = LAMBDA * (activations->acc[board.stm ^ 1][j] > 0.0f);

        local[t].inputWeights[f->features[board.stm][i] * N_HIDDEN + j] += hiddenLosses[board.stm][j]/* + stmLasso*/;
        local[t].inputWeights[f->features[board.stm ^ 1][i] * N_HIDDEN + j] += hiddenLosses[board.stm ^ 1][j]/* + xstmLasso*/;
      }
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->inputWeights[i].g += local[t].inputWeights[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->inputBiases[i].g += local[t].inputBiases[i];
    }
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    for (int t = 0; t < THREADS; t++) {
      g->outputWeights[i].g += local[t].outputWeights[i];
    }
  }

  for (int t = 0; t < THREADS; t++) {
    g->outputBias.g += local[t].outputBias;
  }
}

int main(int argc, char** argv) {
  SeedRandom();

  // Read command options

  int c;

  char nnPath[128] = {0};

  char validPath[128] = {0};
  char trainPath[128] = {0};

  while ((c = getopt(argc, argv, "n:v:t:")) != -1) {
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

      ApplyGradients(nn, gradients, epoch);

      if (((b + 1) % 1000) == 0) {
        printf("Batch: %d / %d\n", b + 1, batches);
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

    printf("Epoch: %3d Valid error: %.8f (%.8f) Time: %ld sec Speed: %7.0f pos/sec\n\n",
           epoch, newValidError, newValidError - validError,
           (epochEndTime - epochStartTime) / 1000,
           1000.0f * trainData->n / (epochEndTime - epochStartTime));

    // Update valid error

    validError = newValidError;
  }

  return 0;
}