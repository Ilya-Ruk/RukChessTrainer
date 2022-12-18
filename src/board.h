#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Feature idx(Piece pc, Square sq, const Color view) {
  if (view == WHITE) {
    return (pc << 6) + sq;
  } else {
    return (opposite[pc] << 6) + (sq ^ 56);
  }
}

INLINE Piece getPiece(Piece pieces[16], int n) {
  return (pieces[n / 2] >> ((n & 1) << 2)) & 0xF;
}

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif