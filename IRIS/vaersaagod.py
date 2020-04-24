import numpy                as np
import matplotlib.pyplot    as plt

## DEFINE
train_start = 00
train_end   = 30
test_start  = 30
test_end    = 50

## TUNE
alpha       = 0.01
NUM_ITER    = 5000

## CHOOSE FEATURES
features = np.array([2])

################################################################################
# Import class data

class_1 = []
class_2 = []
class_3 = []

with open("class_1") as f:
    data = f.readlines()
    for line in data:
        class_1.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
class_1 = np.array(class_1)

with open("class_2") as f:
    data = f.readlines()
    for line in data:
        class_2.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
class_2 = np.array(class_2)

with open("class_3") as f:
    data = f.readlines()
    for line in data:
        class_3.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
class_3 = np.array(class_3)

training_1 = class_1[train_start : train_end, features]
training_2 = class_2[train_start : train_end, features]
training_3 = class_3[train_start : train_end, features]
testing_1  = class_1[test_start : test_end, features]
testing_2  = class_2[test_start : test_end, features]
testing_3  = class_3[test_start : test_end, features]

# train a linear classifier, and tune step factor until training converges.
training = [training_1, training_2, training_3]
testing  = [testing_1, testing_2, testing_3]

################################################################################
def sigmoid(X): # X = W_k*x_ik
    return 1/(1 + np.exp(-X))

################################################################################


C = 3                   # 3 classes
D = len(features)       # D dimensions / number of features
N = len(training[0])    # Training samples of each class
M = len(testing[0])     # Testing samples of each class

W_x = np.zeros((C,D))   # discriminant    # W_x: [3x4]
w_0 = np.zeros((C,1))   # offset          # w_0: [1x3]

W   = np.concatenate((W_x, w_0),axis=1)   # W:   [3x5]
x   = np.zeros((1, D+1))                  # x:   [1x5]

t_k1 = np.array([[1],[0],[0]]) #class1
t_k2 = np.array([[0],[1],[0]]) #class2
t_k3 = np.array([[0],[0],[1]]) #class3
t_k  = [t_k1, t_k2, t_k3]


#training
for m in range(NUM_ITER):
    W_prev      = W
    grad_mse    = np.zeros((C, D+1))

    for k in range(N):   #finding gradient
        for (c, tk) in zip(training, t_k): # c: class   # tk: label for class
            xk          = np.append(c[k], 1)   # making xk a [5x1] matrix
            xk          = xk.reshape(D+1, 1)
            zk          = W@xk       # @ is matrix multiplication in py3.5
            gk          = sigmoid(zk)
            temp        = np.multiply(gk-tk, gk)
            temp        = np.multiply(temp, np.ones((C,1))-gk)
            grad_mse    += temp@xk.T

    norm_grad   = np.linalg.norm(grad_mse)
    W           = W_prev - alpha*grad_mse

np.savetxt('w.txt', W)

#testing
confusion = np.zeros((3,3))
for k in range(M):
    i = 0
    for c in testing: #c = 0, 1,
        g       = W@np.append(c[k], 1)
        g       = g.reshape(C, 1)
        label   = np.argmax(g, axis=0)  # what we think it is
        confusion[i, label] += 1
        i                   += 1

np.savetxt('confusion.txt', confusion)

#error rate:
def error_rate(confusion):
    errors      = 0
    num_samples = 0
    for i in range(C):
        for j in range(C):
            if i != j:
                errors  += confusion[i][j]
            num_samples += confusion[i][j]
    return errors/num_samples

###############################################################################

if __name__== "__main__" :
    print("main. ")
    print(confusion)
    print("error rate: ", error_rate(confusion))
