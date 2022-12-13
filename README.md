# RukChessTrainer
Neural networks (NNUE) trainer (based on [Berserk NN Trainer](https://github.com/jhonnold/berserk-trainer)) for usage with the [RukChess](https://github.com/Ilya-Ruk/RukChess).

# Build
    git clone https://github.com/Ilya-Ruk/RukChessTrainer.git
    cd RukChessTrainer
    cd src
    make

# Usage
1. Create folder c:\Train
2. Download from https://ccrl.chessdom.com/ccrl/404/ archive with games https://ccrl.chessdom.com/ccrl/404/CCRL-404.[2846288].pgn.7z (~2 min., 545.09 Mb)
3. Unzip the archive to c:\Train folder (2846288 games, 3.00 GB)
4. Rename the unpacked file to games.pgn
5. Convert games.pgn to games.fen (use RukChess 3.0, Convert PGN file (games.pgn) to FEN file (games.fen)) (~43 min., 393513458 positions, 21.73 GB)
6. Split (use split_file from [RukChessUtils](https://github.com/Ilya-Ruk/RukChessUtils)) games.fen into validation (10%) and training (90%) data sets (~3 min.)

        split_file.exe games.fen games_valid.fen games_train.fen

7. Create folder c:\Nets
8. Train the neural network (~12 GB RAM, ~535 sec./epoch, ~100 epochs, ~15 hours)

        trainer.exe -v games_valid.fen -t games_train.fen

9. Test different neural network epochs (5 tournaments of 1000 games, ~2 hours/tournament, ~10 hours)

        cutechess-cli.exe -each proto=uci tc=15+0.15 dir="c:\Train" option.Hash=128 option.Threads=1 -engine name="RukChess 3.0 NNUE2" cmd="RukChess 3.0 NNUE2.exe" -engine name="RukChess 3.0 Toga" cmd="RukChess 3.0 Toga.exe" -openings file=book.epd format=epd -repeat -draw movenumber=40 movecount=8 score=20 -resign movecount=3 score=500 -games 2 -rounds 500 -concurrency 6 -ratinginterval 10 -pgnout game_XXX.pgn

10. Choose the best neural network epoch (maximum ELO gain, +287, epoch 40)
11. Determine the level of trained neural network: tournament with programs with known ELO (3 tournaments of 1000 games, ~2 hours/tournament, ~6 hours, ~3169 ELO)

# Resources
1. https://github.com/zamar/spsa
2. https://github.com/cutechess/cutechess
