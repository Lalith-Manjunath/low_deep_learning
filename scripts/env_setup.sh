set +ex

# conda create -p ./env python=3.10 -y
conda activate ./env
pip install torch
python --version
