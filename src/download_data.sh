#!/bin/bash

DATA_DIR=`python -c "from src.config import c; print(c['DATA_DIR'])"`

# competion data

DST_DIR="${DATA_DIR}/competition_data"

if [ ! -d $DST_DIR ]
then
    mkdir -pv $DST_DIR
    cd $DST_DIR
    kaggle competitions download -c plant-pathology-2021-fgvc8
    unzip plant-pathology-2021-fgvc8.zip
    rm plant-pathology-2021-fgvc8.zip
fi

# 2020 data

DST_DIR="${DATA_DIR}/pp_2020"

if [ ! -d $DST_DIR ]
then
    mkdir -pv $DST_DIR
    cd $DST_DIR
    kaggle competitions download -c plant-pathology-2020-fgvc7
    unzip plant-pathology-2020-fgvc7.zip
    rm plant-pathology-2020-fgvc7.zip
fi