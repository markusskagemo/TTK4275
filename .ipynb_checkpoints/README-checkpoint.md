## Classification of handwritten numbers 0-9
NIST is the USA variant of the Norges Forskningsråd (NFR). NIST has designed a database, called
MNIST (see yann.lecun.com/exdb/mnist), with pixtures of handwritten numbers 0-9. The pixtures
have dimension 28x28 pixels and are in 8-bit greyscale; i.e. pixel values between 0-255. For practical
purpose one should note that the the pixtures have been ”preprocessed”; i.e. centred and scaled to
prepare them for classification. Figure 3 shows four ”easy” examples, while figure 4 shows examples
which are harder to classify correctly. A large amount of classifiers have been designed for this case,
resulting in error rates between 1  10 %. The state-of-the-art is (of course) a deep neural network
(DNN).
The database consists of 60000 training examples written by 250 different persons and 10000 test
examples written by 250 other persons.  

The task consists of two parts both using variants of a nearest neighbourhood classifier.

```
pip install tensorflow  
pip install keras
```