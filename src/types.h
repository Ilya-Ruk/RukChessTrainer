#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN 512
#define N_OUTPUT 1

#define THREADS 12
#define BATCH_SIZE 16384

// AdaMax

//#define ALPHA 0.002f
//#define BETA1 0.9f
//#define BETA2 0.999f

// Adam

#define ALPHA 0.01f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

// L1 regularization (Lasso regression)

//#define LAMBDA (1.0 / (1024 * 1024))

#define MAX_VALID_POSITIONS 60000000
#define MAX_TRAIN_POSITIONS 500000000

//#define CRELU_MAX 256

//#define QUANTIZATION_PRECISION_IN 32
//#define QUANTIZATION_PRECISION_OUT 512

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

enum { WHITE, BLACK };

typedef uint8_t Color;
typedef uint8_t Square;
typedef uint8_t Piece;
typedef uint16_t Feature;

typedef struct {
  Color stm;
  uint8_t wdl;
//  Square kings[2];
  uint64_t occupancies;
  uint8_t pieces[16];
} Board; // 28 (32) bytes

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
  float outputWeights[2 * N_HIDDEN] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc1[2][N_HIDDEN] __attribute__((aligned(64)));
} __attribute__((aligned(64))) NNAccumulators;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[2 * N_HIDDEN];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];
} BatchGradients;

extern const float SS;

extern const Piece opposite[12];
//extern const Square psqt[64];

#endif