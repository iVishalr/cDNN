# Contributing

To contribute to cDNN, you need to have a very basic understanding of how neural networks work in general. Having experience in working with libraries like PyTorch or Tensorflow would also help.

You need to also have good understanding of C programming language to make changes in cDNN. I would recommend you to study the documentation first before contributing. Documentation gives you a very brief overview of how cDNN works and also provides a brief explanation of how neural networks work in general.

## Guidelines

1. Avoid commiting directly to the main branch as it will be used by others.
2. When you make changes into the library, please make sure that the library will compile. 
3. After compiling the library, try using it in a .c file and check if there's any memory leaks created due to the changes you have made. This is important because, if you forget to deallocate memory somewhere, the amount of memory usage goes up like 600MB every second. It also depends on the dataset and other things which may vary the memory size.
4. Try overfitting a very small batch of data. A good sanity check would be to get 100% training accuracy by training on a very small data. If we can get 100% accuracy, we can conclude that there's no error in the basic worflow of neural networks.

That's it for now. 
