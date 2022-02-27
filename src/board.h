#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

//INLINE int8_t kIdx(Square k, Square s) { return (k & 4) == (s & 4); }
//INLINE int8_t kIdx(Square k, Square s) { return (((k & 4) == (s & 4)) << 1) + ((k & 32) == (s & 32)); }

INLINE Feature idx(Piece pc, Square sq/*, Square king*/, const Color view) {
  if (view == WHITE) {
    return (pc << 6) + sq;
//    return (pc << 6) + (kIdx(king, sq) << 5) + psqt[sq];
//    return (pc << 7) + (kIdx(king, sq) << 5) + psqt[sq];
  } else {
    return (opposite[pc] << 6) + (sq ^ 56);
//    return (opposite[pc] << 6) + (kIdx(king, sq) << 5) + psqt[sq ^ 56];
//    return (opposite[pc] << 7) + (kIdx(king, sq) << 5) + psqt[sq ^ 56];
  }
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) << 2)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif