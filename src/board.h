#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

//INLINE int8_t kIdx(Square k, Square s) { return (k & 4) == (s & 4); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  if (view == WHITE)
    return (pc << 6) + sq;
  else
    return (inv(pc) << 6) + (sq ^ 56);
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) << 2)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif