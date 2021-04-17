# Dogs VS Cats Dataset Example

In this example, I have showed how `cDNN` can be used to train a simple binary classifier that can differentiate between Cats and Dogs.

Dataset is taken from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

Execute the `Preprocess.ipynb` notebook to preprocess the dataset and extract images and cache it onto your disk.

We can now load the cached dataset into cDNN as shown in `DOGS_VS_CATS.c` and train the model.

You can add more `Dense()` layers and train it with different optimizers and so on. Be careful with the learning rates. When using optimizers like `RMSProp() or Adam()`, a high learning rate could cause the cost to become `nan or inf`. Reduce the learning rate if that happens. `lr = 4.67e-5` was working very well for me.

Note : This will take a long time to train hence I have not fully trained a proper model.
