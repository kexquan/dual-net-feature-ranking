# Dual-net feature ranking
This repository contains codes for dual-net feature ranking (DFR). The codes only work on GPUs.

## Prepare Environment
Activate a new enviroment and install necessary packages:

pip install -r requirements.txt

Tips: If there is any missing package, please refer to 'requirements_full.txt' to install corresponding packages.

## Example 1: XOR synthetic dataset classification
Run example 1 in multithreading, which will take about 5 minutes on a RTX3090 GPU:

python DFR_main.py --run_example1 --operator_arch 128 32 4 --num_fs 5 --multi_thread

## Example 2: binary synthetic dataset classification
Run example 2 in multithreading, which will take about 5 minutes on a RTX3090 GPU:

python DFR_main.py --run_example2 --operator_arch 128 32 2 --num_fs 5 --multi_thread

## Example 3: MNIST hand-written digit feature importance visulization
Run example 3 in multithreading:

python DFR_main.py --run_example3 --num_fs 50 --s 50 --s_p 20 --multi_thread

##
If you find this is useful, please cite 
[Automatic Selection of Discriminative Features for Dementia Detection in Cantonese-Speaking People](http://www.eie.polyu.edu.hk/~mwmak/papers/interspeech22b.pdf)
"Proc. Interspeech 2022"

and Dual-net Feature Ranking and Its Applications to Dementia Detection.

##
Homepage: <https://kexquan.github.io>

Email: xiaoquan.ke@connect.polyu.hk
