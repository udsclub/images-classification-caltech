# images-classification-caltech

This repo will contain all useful links for caltech-256 image classification competition.

## About dataset

- [Info about dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
- [Paper about dataset](https://core.ac.uk/download/pdf/4875878.pdf)

To have same conditions, we split dataset on train and test chunks.
It's up to you get validation set from train one.
We set up kaggle in class competition, but unfortunately it's under review yet.
So splited datasets are available by direct links to S3:

- [train](https://s3-us-west-2.amazonaws.com/usdc-caltech-256/train.zip)
- [test](https://s3-us-west-2.amazonaws.com/usdc-caltech-256/test.zip)
- [sample submission](https://s3-us-west-2.amazonaws.com/usdc-caltech-256/test_example.csv)

In case kaggle in place will not be available, we will create simple HTTP server for validation.

## Info sources

Here are some sources, divided by categories:

### Handling images

- What exactly images are
- Load/crop/resize
- Data augmentation

### Neural networks for images

TODO

### Convolution networks with images

TODO

### Transfer learning

TODO
