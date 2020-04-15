
from numpy import genfromtxt
import matplotlib.pyplot as plt
import platform

if(platform.system() == 'Windows'): #Windows
    class_1 = genfromtxt('class_1', delimiter=',')
    class_2 = genfromtxt('class_2', delimiter=',')
    class_3 = genfromtxt('class_3', delimiter=',')
else:  #Linux
    class_1 = genfromtxt('task_1/Iris_TTT4275/class_1', delimiter=',')
    class_2 = genfromtxt('task_1/Iris_TTT4275/class_2', delimiter=',')
    class_3 = genfromtxt('task_1/Iris_TTT4275/class_3', delimiter=',')


plt.hist(class_1[:,0])
plt.hist(class_2[:,0])
plt.hist(class_3[:,0])
plt.show()
plt.hist(class_1[:,1])
plt.hist(class_2[:,1])
plt.hist(class_3[:,1])
plt.show()
plt.hist(class_1[:,2])
plt.hist(class_2[:,2])
plt.hist(class_3[:,2])
plt.show()
plt.hist(class_1[:,3])
plt.hist(class_2[:,3])
plt.hist(class_3[:,3])
plt.show()
