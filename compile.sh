export CUDA_HOME="/usr/local/cuda"
export CPATH="/usr/local/cuda/include"
export CUDNN_INCLUDE_DIR="/usr/local/cuda/include"
export CUDNN_LIB_DIR="/usr/local/cuda/lib64"

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
cd libs/GANet
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib
