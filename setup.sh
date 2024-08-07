wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -bu
rm Miniconda3-py38_4.12.0-Linux-x86_64.sh


conda create -n ox -c conda-forge --strict-channel-priority osmnx
conda activate ox
pip install -r requirements.txt