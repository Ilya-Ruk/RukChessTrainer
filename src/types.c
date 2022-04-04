#include "types.h"

const float SS = 3.5f / 512;

const Piece opposite[12] = {
	BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING
};

/*const Square psqt[64] = {
	28, 29, 30, 31, 31, 30, 29, 28,
	24, 25, 26, 27, 27, 26, 25, 24,
	20, 21, 22, 23, 23, 22, 21, 20,
	16, 17, 18, 19, 19, 18, 17, 16,
	12, 13, 14, 15, 15, 14, 13, 12,
	 8,  9, 10, 11, 11, 10,  9,  8,
	 4,  5,  6,  7,  7,  6,  5,  4,
	 0,  1,  2,  3,  3,  2,  1,  0
};*/