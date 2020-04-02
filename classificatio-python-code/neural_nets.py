#ref: https://enlight.nyc/projects/neural-network/
import pandas as pd
import numpy as np

class Neural_Network(object):
    def __init__(self):
        self.inputSize=4
        self.outputSize=1
        self.hiddenSize=3
        
        self.W1=np.random.randn(self.inputSize,self.hiddenSize) # 4 x 3
        self.W2=np.random.randn(self.hiddenSize,self.outputSize) # 3 x 1

    def forward(self,X):
        # x size 1 x 4, x here is one row in the database
        self.z = np.dot(X,self.W1) # 1x4 , 4x3 = 1x3
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2) # 1x3, 3x1 = 1x1
        o = self.sigmoid(self.z3)
        return o

    def backward(self,X,Y,o):
        self.o_err=Y-o
        self.o_delta = self.o_err * self.sigmoid_prime(o)

        self.z2_error = self.o_delta.dot(self.W2.T) 
        self.z2_delta = self.z2_error*self.sigmoid_prime(self.z2) 

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

        return sum(0.5*(self.o_err)**2)

    def train(self, X, Y):
        epochs=100
        for k in range(epochs):
            o = self.forward(X)
            loss=self.backward(X, Y, o)
            print("Loss  {}/{} is ".format(k,epochs),np.round(loss,3))
                

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    
    def sigmoid_prime(self,s):
        return s*(1-s)

if __name__ == '__main__':
    # read the input table
    df=pd.read_csv('StudentsDB-numerical.csv')
    # remove the classes colum from the table and convert it to np array
    X=np.array(df.drop(df.columns[[-1]],axis=1),dtype=float)
    # normalize x values to be between 0 and 1
    X_=X/np.amax(X,axis=0)
    X_train=X_[0:14]
    X_test=X_[15:]
    print(X_train)
    # y is the class labels vector
    Y_=np.array(df[df.columns[[-1]]])
    Y_train=Y_[0:14]
    Y_test=Y_[15:]
    NN = Neural_Network()
    # training stage
    NN.train(X_train,Y_train)
    print(NN.forward(X_test) > 0.5)
    #print(X_)