
# This code base is to reproduce Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation  results on different settings
 Refer following code for installation https://github.com/robin-chan/meta-ood

## Pre-requirements
Create a venv and install the packages
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -e detectron2
pip install -r requirements.txt
```
Set environmental variable to store the location of dataset
```
EG: export DETECTRON2_DATASETS=/home/ood/datasets
```
Run the script ```meta-ood/preparation/prepare_coco_segmentation.py``` to prepare the COCO dataset for ood training.
