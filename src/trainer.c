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

    // LOSS CALCULATIONS

    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &board);

    float hiddenLosses[2][N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++) {
      hiddenLosses[board.stm][i] = outputLoss * nn->outputWeights[i] * CReLUPrime(activations->acc[board.stm][i]);
      hiddenLosses[board.stm ^ 1][i] = outputLoss * nn->outputWeights[i + N_HIDDEN] * CReLUPrime(activations->acc[board.stm ^ 1][i]);
    }

    // OUTPUT LAYER GRADIENTS

    local[t].outputBias += outputLoss;

    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].outputWeights[i] += activations->acc[board.stm][i] * outputLoss;
      local[t].outputWeights[i + N_HIDDEN] += activations->acc[board.stm ^ 1][i] * outputLoss;
    }

    // INPUT LAYER GRADIENTS

    for (int i = 0; i < N_HIDDEN; i++) {
//      float stmLasso = LAMBDA * (activations->acc[board.stm][i] > 0);
//      float xstmLasso = LAMBDA * (activations->acc[board.stm ^ 1][i] > 0);

      local[t].inputBiases[i] += hiddenLosses[board.stm][i] + hiddenLosses[board.stm ^ 1][i]/* + stmLasso + xstmLasso*/;
    }

    for (int i = 0; i < f->n; i++) {
      for (int j = 0; j < N_HIDDEN; j++) {
//        float stmLasso = LAMBDA * (activations->acc[board.stm][j] > 0);
//        float xstmLasso = LAMBDA * (activations->acc[board.stm ^ 1][j] > 0);

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

  NN* nn;
  if (!nnPath[0]) {
    printf("No net specified, generating a random net.\n");

    nn = LoadRandomNN();
  } else {
    printf("Loading net from %s\n", nnPath);

    nn = LoadNN(nnPath);
  }

  printf("Loading valid data from %s\n", validPath);

  DataSet* validation = malloc(sizeof(DataSet));
  validation->n = 0;
  validation->entries = malloc(sizeof(Board) * MAX_VALID_POSITIONS);
  LoadEntries(validPath, validation, MAX_VALID_POSITIONS);

  printf("Loading train data from %s\n", trainPath);

  DataSet* data = malloc(sizeof(DataSet));
  data->n = 0;
  data->entries = malloc(sizeof(Board) * MAX_TRAIN_POSITIONS);
  LoadEntries(trainPath, data, MAX_TRAIN_POSITIONS);

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  BatchGradients* local = malloc(sizeof(BatchGradients) * THREADS);

  printf("Calculating Validation Error...\n");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 250; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\n");
    ShuffleData(data);
    printf("Shuffling...DONE\n");

    int batches = data->n / BATCH_SIZE;

    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients, local);
      ApplyGradients(nn, gradients, epoch);

      if ((b + 1) % 1000 == 0) {
        printf("Batch: [#%d/%d]\n", b + 1, batches);
      }
    }

    char buffer[64];
    sprintf(buffer, "../Nets/rukchess_%03d.nnue", epoch);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0 * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;
  }

  return 0;
}