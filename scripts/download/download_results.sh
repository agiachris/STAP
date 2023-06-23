#!/bin/bash

function download_zip {
    gdown "${GDRIVE_LINK}" -O file.zip
    unzip file.zip -d "${DESTINATION}"
    rm file.zip
}

# Download planning results associated with checkpoints to ./plots.
GDRIVE_LINK="https://drive.google.com/uc?id=1aqDms7Rxsfblt2_FnJ_djdj25bO2uLHD"
DESTINATION="./"
download_zip