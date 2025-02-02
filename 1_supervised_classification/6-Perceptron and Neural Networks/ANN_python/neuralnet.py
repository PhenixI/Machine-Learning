﻿#implement the code of an MLP with one input, one hidden, and one output layer to classify the images in the MNIST dataset
import numpy as np
from scipy.special import expit
import sys

class NeuralNetMLP(object):

    #parameter:
    #l2: the lamda parameter for L2 regularization to decrease the degree of overfitting
    #l1: the lamda parameter for L1 regularization
    #epochs�� The Number of passes over the training set.
    #eta: the learning rate eta
    #     a parameter for momentum learning to add a factor of the previous gradient to the weight
    #     update for faster learning deltaW(cur) = eta*deltaJ(Wcur) + alpha * Wpre
    #decrease_const: The decrease constant d for an adaptive learning rate n that decrease
    #                over time for better convergence eta/(1+txd)
    #shuffle: Shuffling the trainging set prior to every epoch to prevent the algorithm from 
    #         getting stuck in cycles
    #Minibatches: Splitting of the training data into k mini-batches in each epoch.
    #             The gradient is computed for each mini-batch separately instead of the 
    #             entire training data for faster learning
     
    def __init__(self,n_output,n_features,n_hidden = 30,l1 = 0.0,l2=0.0,epochs = 500,eta = 0.001,
                 alpha = 0.0,decrease_const = 0.0,shuffle = True,
                 minibatches = 1,random_state = None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1,self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self,y,k):
        onehot = np.zeros((k,y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val,idx] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0,1.0,size = self.n_hidden * (self.n_features+1))
        w1 = w1.reshape(self.n_hidden,self.n_features+1)
        w2 = np.random.uniform(-1.0,1.0,size = self.n_output * (self.n_hidden+1))
        w2 = w2.reshape(self.n_output,self.n_hidden+1)
        return w1,w2

    def _sigmoid(self,z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self,z):
        sg = self._sigmoid(z)
        return sg*(1-sg)

    def _add_bias_unit(self,X,how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0],X.shape[1]+1))
            X_new[:,1:] = X
        elif how =='row':
            X_new = np.ones((X.shape[0]+1,X.shape[1]))
            X_new[1:,:] = X
        else:
            raise AttributeError('how must be column or row')
        return X_new



    def _feedforward(self,X,w1,w2):
        a1 = self._add_bias_unit(X,how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2,how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)

        return a1,z2,a2,z3,a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2)+ np.sum(w2[:, 1:] ** 2))
    
    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum()+ np.abs(w2[:, 1:]).sum())

	#logistic cost function
    def _get_cost(self,y_enc,output,w1,w2):
        term1 = -y_enc*(np.log(output))
        term2 = (1-y_enc)*np.log(1-output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1,w1,w2)
        L2_term = self._L2_reg(self.l2,w1,w2)

        cost = cost +L1_term+L2_term
        return cost
		
	#backpropagation algorithm works to update the weights in Our MLP model
    def _get_gradient(self,a1,a2,a3,z2,y_enc,w1,w2):
        #backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2,how='row')
        sigma2 = w2.T.dot(sigma3)*self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:,:]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        #regularize
        grad1[:,1:] += (w1[:,1:] * (self.l1 +self.l2))
        grad2[:,1:] += (w2[:,1:] * (self.l1 + self.l2))

        return grad1,grad2

    def predict(self,X):
        a1,z2,a2,z3,a3 = self._feedforward(X,self.w1,self.w2)
        y_pred = np.argmax(z3,axis=0)
        return y_pred

    def fit(self,X,y,print_progress=False):
        #save the cost for each epoch in a cost_ list that can visualize
        self.cost_ = []
        X_data,y_data = X.copy(),y.copy()
        y_enc = self._encode_labels(y,self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            #adaptive learning rate
            self.eta /= (1+self.decrease_const*i)

            if print_progress:
                sys.stderr.write('\r Epoch: %d/%d' % (i+1,self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data,y_data = X_data[idx],y_data[idx]

            mini = np.array_split(range(y_data.shape[0]),self.minibatches)

            for idx in mini:
                #feedforward
                a1,z2,a2,z3,a3 = self._feedforward(X[idx],self.w1,self.w2)
                cost = self._get_cost(y_enc = y_enc[:,idx],output = a3,w1 = self.w1,w2 = self.w2)

                self.cost_.append(cost)

                #compute gradient via backpropagation
                grad1,grad2 = self._get_gradient(a1=a1,a2=a2,
                                                 a3=a3,z2=z2,
                                                 y_enc = y_enc[:,idx],
                                                 w1 = self.w1,
                                                 w2 = self.w2)

                ##start gradient checking
                #grad_diff = self._gradient_checking(X=X[idx],
                #                                    y_enc = y_enc[:,idx],
                #                                    w1=self.w1,
                #                                    w2=self.w2,
                #                                    epsilon = 1e-5,
                #                                    grad1=grad1,
                #                                    grad2 = grad2)

                #if grad_diff <= 1e-7:
                #    print('Ok: %s' % grad_diff)
                #elif grad_diff <= 1e-4:
                #    print('Warning: %s' % grad_diff)
                #else:
                #    print('Problem %s' %grad_diff)

                ##end gradient checking



                #update weights;[alpha *delta_w_prev] for momentum learning
                delta_w1,delta_w2 = self.eta * grad1,self.eta * grad2
                self.w1 -= (delta_w1+(self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 +(self.alpha * delta_w2_prev))

                delta_w1_prev,delta_w2_prev = delta_w1,delta_w2

        return self

    def _gradient_checking(self,X,y_enc,w1,w2,epsilon,grad1,grad2):
        """Apply gradient checking (for debugging only)
        Returns
        -----------------------
        relative _error :float
            Relative error between the numerically 
            approximated gradients and the backpropagated gradients
        """
        num_grad1 = np.zeros(np.shape(w1))
        epsilon_ary1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ary1[i,j] = epsilon
                a1,z2,a2,z3,a3 = self._feedforward(X,w1-epsilon_ary1,w2)
                cost1 = self._get_cost(y_enc,a3,w1-epsilon_ary1,w2)
                a1,ze,a2,z3,a3 = self._feedforward(X,w1+epsilon_ary1,w2)
                cost2 = self._get_cost(y_enc,a3,w1+epsilon_ary1,w2)

                num_grad1[i,j] = (cost2-cost1)/(2*epsilon)
                epsilon_ary1[i,j] = 0

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))

        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ary2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X,w1,w2 - epsilon_ary2)
                cost1 = self._get_cost(y_enc,a3,w1,w2 - epsilon_ary2)
                a1, z2, a2, z3, a3 = self._feedforward(X,w1,w2 + epsilon_ary2)
                cost2 = self._get_cost(y_enc,a3,w1,w2 + epsilon_ary2)
                
                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary2[i, j] = 0
                
        num_grad = np.hstack((num_grad1.flatten(),num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1 / (norm2 + norm3)
        
        return relative_error



