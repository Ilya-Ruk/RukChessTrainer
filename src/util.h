#ifndef UTIL_H
#define UTIL_H

#define INLINE static inline __attribute__((always_inline))

long GetTimeMS(void);

void* AlignedMalloc(int size);
void AlignedFree(void* ptr);

#endif