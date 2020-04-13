

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import platform
import itertools

if(platform.system() == 'Windows'): #Windows
    class_1 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_1', delimiter=',')
    class_2 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_2', delimiter=',')
    class_3 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_3', delimiter=',')
else:  #Linux
    class_1 = genfromtxt('task_1/Iris_TTT4275/class_1', delimiter=',')
    class_2 = genfromtxt('task_1/Iris_TTT4275/class_2', delimiter=',')
    class_3 = genfromtxt('task_1/Iris_TTT4275/class_3', delimiter=',')


##############################
##extracting wanted features##
features=np.array([1,2,3]) #if you want to change features->CHANGE ME
class_1=class_1[:,features]
class_2=class_2[:,features]
class_3=class_3[:,features]
num_f=features.shape[0]

###
train_begin=0
train_end=30
####
test_begin=30
test_end=50
####

W = np.ones((3,num_f+1))*0.01
tk_1 = np.zeros((3,1))
tk_1[0,0] = 1
tk_2 = np.zeros((3,1))
tk_2[1,0] = 1
tk_3 = np.zeros((3,1))
tk_3[2,0] = 1
norm_grad_MSE = 100
alpha = 0.01

def sigmoid(X):
   return 1/(1+np.exp(-X))

for i in range(5000):
    N=30 #how many to train on
    grad_MSE = np.zeros((3,num_f+1))
    for n in range(train_begin,train_end):
        for (_class,tk) in zip([class_1,class_2,class_3],[tk_1,tk_2,tk_3]):
            xk = np.append(_class[n,:],1).reshape((num_f+1,1))
            zk = W@xk
            gk = sigmoid(zk)
            temp = np.multiply(gk-tk,gk)
            temp = np.multiply(temp, np.ones((3,1))-gk)
            grad_MSE += alpha*temp@xk.T
    W -= grad_MSE
    norm_grad_MSE = np.linalg.norm(grad_MSE)
np.savetxt('w.txt',W)
W = np.loadtxt('w.txt')

###############################
### Classification ################
confusion = np.zeros([3,3])
for n in range(test_begin,test_end):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        g = W@np.append(class_[n,:],1).reshape(num_f+1,1)
        classified_class = np.argmax(g, axis=0)
        confusion[i,classified_class] += 1
        i += 1

print(confusion)


############################
###error-rate##
def find_error_rate(confusion):
    N=np.sum(confusion)
    error_counter=0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[0]):
            if(i != j):
                error_counter += confusion[i,j]
    return(error_counter/N)

print(f'Error rate:{find_error_rate(confusion)}')
