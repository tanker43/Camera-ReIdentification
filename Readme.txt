Installation

Make sure conda is installed.

# cd to your preferred directory and clone this repo
git clone https://github.com/KaiyangZhou/deep-person-reid.git

# create environment
cd deep-person-reid/
conda create --name torchreid python=3.7
conda activate torchreid

# install dependencies
(e.g:  pip install numpy)

numpy
Cython
h5py
Pillow
six
scipy
opencv-python
matplotlib
tb-nightly
future
yacs
gdown
flake8
yapf
isort==4.3.21
imageio



# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# install torchreid (don't need to re-build it if you modify the source code)
python setup.py develop


To train OSNet on Market1501, do

python scripts/main.py 

--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml 
--transforms random_flip random_erase 
--root $PATH_TO_DATA (Assume $PATH_TO_DATA is the directory containing reid datasets. )