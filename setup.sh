CONDA_PATH=$1
CONDA_ENV_NAME=$2

# activate conda environment
source ${CONDA_PATH}/bin/activate
source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME}

# install Pytorch 2.0 with Cuda 11.7
yes | pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# install nerfstudio
yes | pip install --upgrade nerfstudio==0.3.4

# install pytorch3d
yes | pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install other nerfstudio extension packages
yes | pip install --upgrade pycocotools
yes | pip install --upgrade pixellib
yes | pip install --upgrade torchtyping
yes | pip install --upgrade plyfile

# install utils for point cloud management
yes | pip install --upgrade git+https://github.com/lxxue/prefix_sum.git
yes | pip install --upgrade git+https://github.com/lxxue/FRNN.git
yes | pip install --upgrade torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
yes | pip install --upgrade torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# install lpips
yes | pip install lpips

# install simmple knn
pip install git+https://github.com/camenduru/simple-knn.git@44f7642

# install splatting with nerfstudio, version 1.1.0
pip install git+https://github.com/nerfstudio-project/gsplat.git@9979ed6

# install additional dependencies
yes | pip install gdown
