#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN 512
#define N_OUTPUT 1

#define THREADS 12
#define BATCH_SIZE 16384

#define NUM_REGS (N_HIDDEN * (int)sizeof(float) / (int)sizeof(__m256)) // 64

// Adam

#define ALPHA 0.002f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

// Limits

#define MAX_VALID_POSITIONS 10000000
#define MAX_TRAIN_POSITIONS 450000000

#define MAX_EPOCHS 500

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
  float outputWeights[N_HIDDEN * 2] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc[2][N_HIDDEN] __attribute__((aligned(64)));
} __attribute__((aligned(64))) NNAccumulators;

typedef struct {
  float M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[N_HIDDEN * 2];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN * 2];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];
} BatchGradients;

extern const float SS;

extern const Piece opposite[12];

#endif