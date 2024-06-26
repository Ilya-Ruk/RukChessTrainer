#ifndef BITS_H
#define BITS_H

#include <inttypes.h>

#include "types.h"
#include "util.h"

#define bit(sq) (1ULL << (sq))
#define setBit(bb, sq) ((bb) |= bit(sq))
#define lsb(bb) (__builtin_ctzll(bb))

INLINE Square popLsb(uint64_t* bb)
{
  Square sq = lsb(*bb);

  *bb &= *bb - 1;

  return sq;
}

#endif