conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio
CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-scatter # -f https://data.pyg.org/whl/torch-1.14.0+${cpu}.html
CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-sparse #  -f https://data.pyg.org/whl/torch-1.14.0+${cpu}.html
CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-geometric
