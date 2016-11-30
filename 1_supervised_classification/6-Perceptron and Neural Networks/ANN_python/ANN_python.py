#Classifying handwritten digits

#1.obtaining the mnist dataset
import os
import struct
import numpy as np

#1.the frst being an n m �� dimensional
#NumPy array (images), where n is the number of samples and m is the number
#of features

#2.The second array (labels) returned by the load_mnist
#function contains the corresponding target variable, the class labels (integers 0-9) of
#the handwritten digits.

def load_mnist(path,kind = 'train'):
    """Load Mnist data from path"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)

    with open(labels_path,'rb') as lbpath:
        #read 8 byte ,each 4 byte as unsigned integer
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype = np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack(">IIII",imgpath.read(16))

        images = np.fromfile(imgpath,dtype = np.uint8).reshape(len(labels),784)

    return images,labels

#load the 60,000 training instances as well as the 10,000 test samples
X_train,y_train = load_mnist('F:\developSamples\ml\mnist',kind = 'train')

print ('Rows: %d,columns: %d' % (X_train.shape[0],X_train.shape[1]))

X_test,y_test = load_mnist('F:\developSamples\ml\mnist',kind='t10k')
print('Rows %d,columns: %d' % (X_test.shape[0],X_test.shape[1]))

#visualize examples of the digits 0-9 after reshaping the 784-pixel vectors
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True,sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28,28)
    ax[i].imshow(img,cmap = 'Greys',interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#plot multiple examples of the same digit
fig,ax = plt.subplots(nrows = 5,ncols = 5,sharex = True,sharey = True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28,28)
    ax[i].imshow(img,cmap = 'Greys',interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

import os
os.chdir('E:/machine-learning/1_supervised_classification/6-Perceptron and Neural Networks/ANN_python')
from neuralnet import NeuralNetMLP

#a 784-50-10 MLP
nn = NeuralNetMLP(n_output = 10,
                  n_features = X_train.shape[1],
                  n_hidden = 50,
                  l2 = 0.1,
                  l1 = 0.0,
                  epochs = 1000,
                  eta = 0.001,
                  alpha = 0.001,
                  decrease_const = 0.00001,
                  shuffle = True,
                  minibatches = 50,
                  random_state = 1)

nn.fit(X_train,y_train,print_progress = True)

#plot every 50th step to account for the 50 mini-batches (50 mini-batches x1000 epochs) 
plt.plot(range(len(nn.cost_)),nn.cost_)
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs *50')
plt.tight_layout()
plt.show()

#plot a smoother version of the cost function against the number of epochs by averaging over the mini-batch
#intervals
batches = np.array_split(range(len(nn.cost_)),1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)),cost_avgs,color = 'red')
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

#evaluate the performance of the model by calculating the prediction accuracy
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred,axis = 0)/X_train.shape[0]
print ('Training accuracy %.2f%%' % (acc*100))

#generalize to data that it has not seen before
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred,axis = 0)/X_test.shape[0]
print ('Training accuracy %.2f%%' % (acc*100))

#plot the images that MLP struggles with
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig,ax = plt.subplots(nrows = 5,
                      ncols = 5,
					  sharex = True,
					  sharey = True,)

ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation = 'nearest')
    ax[i].set_title(' %d) t:%d p: %d' % (i+1,correct_lab[i],miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#gradient checking example

nn = NeuralNetMLP(n_output = 10,
                  n_features = X_train.shape[1],
                  n_hidden = 10,
                  l2 = 0.0,
                  l1 = 0.0,
                  epochs = 10,
                  eta = 0.001,
                  alpha = 0.0,
                  decrease_const = 0.0,
                  minibatches = 1,
                  random_state = 1)

nn.fit(X_train[:5],y_train[:5],print_progress = False)


	






