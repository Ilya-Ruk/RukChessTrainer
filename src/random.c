#include "random.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "util.h"

// https://prng.di.unimi.it/splitmix64.c

static uint64_t state; /* The state can be seeded with any value. */

static uint64_t SplitMix64(void)
{
  uint64_t z = (state += 0x9e3779b97f4a7c15);

  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;

  return z ^ (z >> 31);
}

// https://prng.di.unimi.it/xoshiro256plusplus.c

static uint64_t s[4];

INLINE uint64_t rotl(const uint64_t x, int k)
{
  return (x << k) | (x >> (64 - k));
}

uint64_t RandomUInt64(void)
{
  const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

  const uint64_t t = s[1] << 17;

  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];

  s[2] ^= t;

  s[3] = rotl(s[3], 45);

  return result;
}

void SeedRandom(void)
{
  state = time(NULL);

  s[0] = SplitMix64();
  s[1] = SplitMix64();
  s[2] = SplitMix64();
  s[3] = SplitMix64();
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/

float RandomGaussian(float mu, float sigma)
{
  float U1, U2, W, mult;

  static float X1, X2;
  static int call = 0;

  if (call == 1) {
    call = !call;

    return (mu + sigma * X2);
  }

  do {
    U1 = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;
    U2 = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;

    W = powf(U1, 2.0f) + powf(U2, 2.0f);
  } while (W == 0.0f || W >= 1.0f);

  mult = sqrtf((-2.0f * logf(W)) / W);

  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * X1);
}