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
        # X size 1 x 4, x here is one row in the database
        self.z = np.dot(X,self.W1) # 1x4 , 4x3 = 1x3
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2) # 1x3, 3x1 = 1x1
        o = self.sigmoid(self.z3)
        print(o)

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

if __name__ == '__main__':
    # read the input table
    df=pd.read_csv('StudentsDB-numerical.csv')
    # remove the classes colum from the table and convert it to np array
    X=np.array(df.drop(df.columns[[-1]],axis=1),dtype=float)
    # normalize x values to be between 0 and 1
    x=X/np.amax(X,axis=0)
    # y is the class labels vector
    y=np.array(df[df.columns[[-1]]])
    NN = Neural_Network()
    NN.forward(x[0])
    #print(X_)