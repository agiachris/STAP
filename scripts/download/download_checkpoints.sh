#!/bin/bash

function download_zip {
    gdown "${GDRIVE_LINK}" -O file.zip
    unzip file.zip -d "${DESTINATION}"
    rm file.zip
}

# Download model checkpoints and IRL dataset to ./models.
GDRIVE_LINK="https://drive.google.com/uc?id=1QeAUF73zZSKOA1vhMEFZ9KU4BrNv0eVI"
DESTINATION="./"
download_zip

# Download environment configs associated with IRL dataset.
# to ./configs/pybullet/envs/official/primitives/datasets.
GDRIVE_LINK="https://drive.google.com/uc?id=1uxm4TZq0XSS2dk5mp8pIz3Q6AGkSLK8C"
DESTINATION="./configs/pybullet/envs/official/primitives/"
download_zip