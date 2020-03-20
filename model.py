from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
#from utils import euclidean_distance_loss


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def train_keras_model(X_train, Y_train):
    # Init keras model
    model = Sequential()
    # Specify model structure
    model.add(Dense(output_dim=64, input_dim=100)) # > fiks parametre
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10)) # Ti klasser
    model.add(Activation("softmax"))
    
    # Compile model
    model.compile(loss=euclidean_distance_loss, 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    
    # Train the neural network
    model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
    
    return model
