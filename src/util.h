#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

#include "types.h"

#define INLINE static inline __attribute__((always_inline))

long GetTimeMS(void);

void* AlignedMalloc(int size);
void AlignedFree(void* ptr);

#endif
