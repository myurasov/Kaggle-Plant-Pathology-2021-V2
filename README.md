# Code for Kaggle Plant Pathology 2021 - FGVC8

https://www.kaggle.com/c/plant-pathology-2021-fgvc8

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`

## Plan

- Remove duplicates
- Split into N folds equally balancing rare classes
- Train N models with augmentation
- Average weights (average predictions?)

Ideas:

- Detect outliers behaving differently in some folds. Correct labels? Adjust sample weights?
- Freeze base model in the beginnging of each run to allow additions to adjust
- Average last N model weights

## Running

### Prepare Dataset

`$ docker/docker.sh src/download_data.sh`
`$ docker/docker.sh "src/prepare_dataset.run.py --folds 5"`
`$ docker/docker.sh "src/drop_duplicates.run.py"`
`$ docker/docker.sh "src/cache_images.run.py --size W H"`

### Train

`$ docker/docker.sh src/train.run.py`

## Links

- https://www.kaggle.com/c/plant-pathology-2021-fgvc8
