from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
import keras.backend as K


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def train_keras_ann(X_train, Y_train, n_epochs=10):
    model = Sequential()
    # Specify model structure
    model.add(Dense(units=32, input_dim=784)) # > fiks parametre
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=64))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=128))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=10)) # Ti klasser
    model.add(Activation("softmax"))
    
    # Compile model
    """
    model.compile(loss=euclidean_distance_loss, 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    """
    model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    
    # Train the neural network
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=32)
    
    return model


def train_keras_conv(X_train, Y_train, n_epochs=10):
    #https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
    model = Sequential()
    # Convolutional part
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    # ANN part
    model.add(Flatten())
    model.add(Dense(units=10))
    #model.add(Dropout(0.2))
    model.add(Activation("softmax"))
    
    # Compile model
    model.compile(loss=euclidean_distance_loss, 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    
    # Train the neural network
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=32)
    
    return model
