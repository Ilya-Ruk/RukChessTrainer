#include "board.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bits.h"
#include "types.h"

void ToFeatures(Board* board, Features* f)
{
  f->n = 0;

  uint64_t bb = board->occupancies;

  while (bb) {
    Square sq = popLsb(&bb);
    Piece pc = getPiece(board->pieces, f->n);

    f->features[WHITE][f->n] = idx(pc, sq, WHITE);
    f->features[BLACK][f->n] = idx(pc, sq, BLACK);

    f->n++;
  }
}

void ParseFen(char* fen, Board* board)
{
  char* _fen = fen;
  int n = 0;

  // Make sure the board is empty

  board->occupancies = 0ULL;

  for (int i = 0; i < 16; i++) {
    board->pieces[i] = 0;
  }

  for (Square sq = 0; sq < 64; sq++) {
    char c = *fen;

    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      Piece pc;

      switch (c) {
        case 'P':
          pc = WHITE_PAWN;
          break;

        case 'N':
          pc = WHITE_KNIGHT;
          break;

        case 'B':
          pc = WHITE_BISHOP;
          break;

        case 'R':
          pc = WHITE_ROOK;
          break;

        case 'Q':
          pc = WHITE_QUEEN;
          break;

        case 'K':
          pc = WHITE_KING;
          break;

        case 'p':
          pc = BLACK_PAWN;
          break;

        case 'n':
          pc = BLACK_KNIGHT;
          break;

        case 'b':
          pc = BLACK_BISHOP;
          break;

        case 'r':
          pc = BLACK_ROOK;
          break;

        case 'q':
          pc = BLACK_QUEEN;
          break;

        case 'k':
          pc = BLACK_KING;
          break;

        default:
          printf("Unable to parse FEN: %s!\n", _fen);
          exit(1);
      }

      setBit(board->occupancies, sq);

      board->pieces[n / 2] |= pc << ((n & 1) << 2);

      n++;
    } else if (c >= '1' && c <= '8') {
      sq += (c - '1');
    } else if (c == '/') {
      sq--;
    } else {
      printf("Unable to parse FEN: %s!\n", _fen);

      exit(1);
    }

    fen++;
  }
}