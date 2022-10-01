#include "random.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// https://prng.di.unimi.it/splitmix64.c

static uint64_t state; /* The state can be seeded with any value. */

uint64_t next(void) {
	uint64_t z = (state += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

// https://prng.di.unimi.it/xoshiro256plusplus.c

static uint64_t s[4];

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t RandomUInt64(void) {
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

void SeedRandom(void) {
	state = time(NULL);

	s[0] = next();
	s[1] = next();
	s[2] = next();
	s[3] = next();
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
float RandomGaussian(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1) {
    call = !call;
    return (mu + sigma * (float)X2);
  }

  do {
    U1 = -1 + ((float)rand() / RAND_MAX) * 2;
    U2 = -1 + ((float)rand() / RAND_MAX) * 2;
    W = pow(U1, 2) + pow(U2, 2);
  } while (W >= 1 || W == 0);

  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (float)X1);
}
