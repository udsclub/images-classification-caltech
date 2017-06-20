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

## Roadmap

### In case you are new to the machine learning

Unfortunately this competition is not intended to cover machine learning from the beginning.
In case you still want to participate try to cover required topics from [this blog post](https://medium.com/towards-data-science/howto-became-a-computer-scientist-2ecb6e9e7835) as fast as possible and find some sub mentor from participants.

### In case you are new to the images processing

If you don't know how images are represented in python briefly cover one of such tutorials:

- [pillow tutorial](http://pillow.readthedocs.io/en/3.1.x/handbook/tutorial.html) - one of the default python library for image processing, but nowadays used not very often. Posted here only for reference.
- [numpy and scipy images processing](http://www.scipy-lectures.org/advanced/image_processing/) - as per my opinion preferred one. Will explain you how to handle images as numpy array and basic manipulation. If you don't know hot to use numpy - take this [numpy tutorial](http://www.scipy-lectures.org/intro/numpy/index.html). Numpy is de facto standart python library for handling matrices.
- [Scikit-image: image processing](http://www.scipy-lectures.org/packages/scikit-image/) - advanced tutorial, that will show how you may manually process your images.

### Prepare dataset

After you know how to handle images it'a time to prepare dataset. Of course you may use framework based decisions(later about them) but I highly advice to create your own preprocessor. It should generate images with same size. It's up to you how it should be done - they can be resized, cropped or filled with some noise. After resize it's better to store them separately - this will reduce computation time during training. [Here is](https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network) quite good discussion what images preprocessing can be done(check first answer).

### Choose framework

Here is the brief list of neural networks(NN) frameworks. You can use not NN algorithms also, but I out of this, so I will describe them. Also, *this is personal opinion*, don't rise holywars here. If there are will be a lot of complains, we may change this part.

- [mxnet](http://mxnet.io/) - as per me quite strange library with hard to explain tracebacks. Don't have a lot of experience with it, but already don't like it.
- [caffe](http://caffe.berkeleyvision.org/) - quite popular framework for images processing. Have some pretrained models. I have no any expertise in it, but it should be covered by a lot of tutorials, because it quite old.
- [theano](http://deeplearning.net/software/theano/) - one more framework. I can only say that I've rewrite a lot of code from it to tensorlfow :)
- [tensorlow](https://www.tensorflow.org/) - google framework for ML. Have a lot of stuff out of the box. But also can be quite mess because of a lot of available features. Have a lot of tutorials. I've code with it about a year, so I can help with it.
- [keras](https://keras.io/) - syntactic sugar above theano or tensorflow. With it you can write your own model quite fast. Out of the box have all stuff that exist in theano and tensorflow, and a lot of self provided additional features, as data loader for example. Easy to understand. **I highly advice to use keras with tensorflow backend** is you want to train the NN, and not spend about a 2 weeks reading tons of manuals.
- [pytorch](http://pytorch.org/) - new ML framework, inherited from [torch](http://torch.ch/). It's very young, so I cannot say anything. As first opinion it's have clear structure, a little bit strange approach to the optimization and some bugs. 

### Neural networks for images

TODO

### Convolution networks with images

TODO

### Transfer learning

TODO
