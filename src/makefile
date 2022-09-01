CC = gcc
SRC = *.c
EXE = ../bin/trainer

LIBS = -lm
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -fopenmp -march=skylake

all:
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE)