#ifndef RANDOM_H
#define RANDOM_H

#include <inttypes.h>

uint64_t RandomUInt64(void);
void SeedRandom(void);

float RandomGaussian(float mu, float sigma);

#endif