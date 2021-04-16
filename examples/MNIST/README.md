# MNIST Dataset Example

As an example for cDNN, I have tired to show how cDNN can be used to train a model that recognises handwritten digits. 

I have trained a descent model that achieves a 90% accuracy on test set (10k images, all passed at once), 93.90% accuracy in validation set and 97.64% accuracy on the training set.

The learning rate was decreased by a factor of 10 whenever the val_acc reached a plateau.

SGD + Momentum was used to train the model with momentum=0.9 and initial lr = 4.67e-3.

L2 Regualrization was used with a weight_decay of 5e-4

Batch_size used is 256

Training Set : 55,000 images
Validation Set : 5000 images
Test Set : 10,000 images.

Data set is available to download [here](http://yann.lecun.com/exdb/mnist/).

The data set has been preprocess using the `Preprocess.ipynb` notebook and train,validation and test sets were created and stored in their respective files.