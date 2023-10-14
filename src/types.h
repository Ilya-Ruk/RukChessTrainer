#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN_1 256
#define N_HIDDEN_2 16
#define N_OUTPUT 1

#define THREADS 12
#define BATCH_SIZE 16384

#define CRELU_MAX 1.0f

#define NUM_REGS_1 (N_HIDDEN_1 * (int)sizeof(float) / (int)sizeof(__m256)) // 64
#define NUM_REGS_2 (N_HIDDEN_2 * (int)sizeof(float) / (int)sizeof(__m256)) // 4

// Adam

#define ALPHA 0.002f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define MAX_VALID_POSITIONS 60000000
#define MAX_TRAIN_POSITIONS 500000000

// Dropout

//#define DROPOUT
//#define DROPOUT_P 0.5f // 50%
//#define DROPOUT_Q (1.0f - DROPOUT_P)

enum {
  WHITE_PAWN,
  WHITE_KNIGHT,
  WHITE_BISHOP,
  WHITE_ROOK,
  WHITE_QUEEN,
  WHITE_KING,
  BLACK_PAWN,
  BLACK_KNIGHT,
  BLACK_BISHOP,
  BLACK_ROOK,
  BLACK_QUEEN,
  BLACK_KING,
};

enum {
  WHITE,
  BLACK
};

typedef uint8_t Color;
typedef uint8_t Square;
typedef uint8_t Piece;
typedef uint16_t Feature;

typedef struct {
  Color stm;
  uint8_t wdl;
  uint64_t occupancies;
  Piece pieces[16];
} Board; // 26 (32) bytes

typedef struct {
  int8_t n;
  Feature features[2][32];
} Features;

typedef struct {
  int n;
  Board* entries;
} DataSet;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_2] __attribute__((aligned(64)));

  float hiddenBiases[N_HIDDEN_2] __attribute__((aligned(64)));
  float hiddenWeights[N_HIDDEN_1 * N_HIDDEN_2 * 2] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN_1] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN_1] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc2[N_HIDDEN_2] __attribute__((aligned(64)));
  float acc1[2][N_HIDDEN_1] __attribute__((aligned(64)));
} __attribute__((aligned(64))) NNAccumulators;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[N_HIDDEN_2];

  Gradient hiddenBiases[N_HIDDEN_2];
  Gradient hiddenWeights[N_HIDDEN_1 * N_HIDDEN_2 * 2];

  Gradient inputBiases[N_HIDDEN_1];
  Gradient inputWeights[N_INPUT * N_HIDDEN_1];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_2];

  float hiddenBiases[N_HIDDEN_2];
  float hiddenWeights[N_HIDDEN_1 * N_HIDDEN_2 * 2];

  float inputBiases[N_HIDDEN_1];
  float inputWeights[N_INPUT * N_HIDDEN_1];
} BatchGradients;

extern const float SS;

extern const Piece opposite[12];

#endif