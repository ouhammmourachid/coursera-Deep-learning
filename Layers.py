# libraries that we will need are
from re import A
import numpy as np
from numpy import float32,array
EPSILON = 10e-15

# definition of the super class layer

class _Layer:
    def __init__(self,n_l:int) -> None:
        self.n_l = n_l
    def linear_activation_forward(self) -> None:
        pass
    def update(self,learning_rate:float) -> None:
        pass
    def norm_weight(self) -> float:
        return 0
    def initialize_weights(self) -> None:
        pass


# definition of input calss:

class InputLayer(_Layer):
    def __init__(self,n_l = None) -> None:
        _Layer.__init__(self,n_l)
        self.A = None
    def linear_activation_forward(self,X:array) -> array:
        self.A = X
        return self.A
    def flatten_y(self,y:array) -> array:
        return y


# flatten layer:

class Flatten(InputLayer):
    def __init__(self,input_shape:list) -> None:
        InputLayer.__init__(self,np.prod(np.array(input_shape)))

    # the exprition to flateen X is X.reshape(X.shape[0],-1).T
    def linear_activation_forward(self,X:array) -> array:
        return InputLayer.linear_activation_forward(self,X.reshape(X.shape[0],-1).T)
            

    # the exprition to flateen y is y.T

    def flatten_y(self,y:array) -> array:
        return y.T


# normalization layer:


class Normalization(Flatten):
    def __init__(self,input_shape:list)->None:
        Flatten.__init__(self,input_shape)
            
    def linear_activation_forward(self,X:array)->array:
        Flatten.linear_activation_forward(self,X)
        self.A = np.divide(self.A,np.linalg.norm(self.A,axis=0,keepdims=True))
        return self.A


# standarization layer:


class Standarization(Flatten):
    def __init__(self,input_shape:list) -> None:
        Flatten.__init__(self,input_shape)

    def linear_activation_forward(self,X:array)->array:
        Flatten.linear_activation_forward(self,X)
        self.A = Standarization.standrlise(self.A)
        return self.A
    
    def standrlise(X:array)->array:
        epsilon = 10E-8
        X_mean = np.mean(X,axis=0,keepdims=True)
        X_var = np.var(X,axis=0,keepdims=True)
        X = np.divide(X-X_mean,np.sqrt(X_var+epsilon))
        return X

class _ActivationDense(_Layer):
    def __init__(self, activation : str,n_l = None) -> None:
        super().__init__(n_l)
        self.dA = None
        self.A = None
        self.Z = None
        self.activation = None
        self.backward_activation = None
        self.initialize_activation(activation)
    
    # soft max function:
    def softmax(z:array)->array:
        exp_prob = np.exp(z)
        return np.divide(exp_prob,np.sum(exp_prob,axis=0,keepdims=True))
    # sigmoid function:
    def sigmoid(z:array) -> array:
        return 1/(1+np.exp(-z))
    # relu function:
    def relu(z:array)->array:
        return np.maximum(0,z)
    # tanh function:
    def tanh(z:array)->array:
        return np.tanh(z)
    # deivative of softmax:
    def softmax_derv(z:array) -> array:
        pass
    # derivative of sigmoid:
    def sigmod_derv(z:array) -> array:
        return Activation.sigmoid(z) - Activation.sigmoid(z)**2
    # derivative of relu:
    def relu_derv(z:array) -> array:
        return np.array(z > 0,dtype=float32)
    # derivative of tanh:
    def tanh_derv(z:array) -> array:
        return 1/(np.cosh(z))**2
            

    def initialize_activation(self,activation:str) -> None:
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.backward_activation = self.sigmod_derv
        elif activation == 'relu':
            self.activation = self.relu
            self.backward_activation = self.relu_derv
        elif activation == 'tanh':
            self.activation = self.tanh
            self.backward_activation = self.tanh_derv
        elif activation == 'softmax':
            self.activation = self.softmax
            self.backward_activation = self.softmax_derv
        elif activation == 'linear':
            self.activation = lambda z : z
            self.backward_activation = lambda z : 1

# hidden layer:

class Dense(_ActivationDense):
    def __init__(self,n_l:int,activation = 'linear',use_bias = True) -> None:
        _ActivationDense.__init__(self,activation,n_l)
        self.use_bias = use_bias
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.dZ = None    

    # intialization function for the weight


    def initialize_weights(self,n_l_prev:int,alpha=0.1,type='random')->None:
        if self.use_bias:
            self.b = np.zeros((self.n_l,1))
        if type == 'random':
            self.W = np.random.randn(self.n_l,n_l_prev)*alpha
        elif type == 'Xavier':
            self.W = np.random.randn(self.n_l,n_l_prev)*np.sqrt(2/(self.n_l+n_l_prev))
        elif type =='he':
            self.W = np.random.randn(self.n_l,n_l_prev)*np.sqrt(2/n_l_prev)
        elif type == 'leCun':
            self.W = np.random.randn(self.n_l,n_l_prev)*np.sqrt(1/n_l_prev)

        
    
    def linear_activation_forward(self,A_prev:array)->array:
        self.Z = np.dot(self.W,A_prev) 
        if self.use_bias:
            self.Z = self.Z + self.b
        self.A = self.activation(self.Z)
        return self.A

    def linear_backward(self,A_prev,lambd:float)->array:
        m = A_prev.shape[1]
        self.dW = 1/m*np.dot(self.dZ,A_prev.T)
        if self.use_bias:
            self.db = 1/m*np.sum(self.dZ,axis=1,keepdims=True)
        if lambd != 0 :
            self.dW = self.dW + lambd/m*self.W
            if self.use_bias:
                self.db = self.db + lambd/m*self.b
        return np.dot(self.W.T,self.dZ)

    def linear_activation_backward(self,A_prev:array,lambd:float)->array:
        self.dZ = self.backward_activation(self.Z)
        self.dZ = np.multiply(self.dA,self.backward_activation(self.Z))
        dA_prev = self.linear_backward(A_prev,lambd)
        return dA_prev
    
    def update(self,learning_rate:float) -> None:
        assert(self.W.shape == self.dW.shape)
        self.W = self.W -learning_rate * self.dW
        if self.use_bias:
            assert(self.b.shape == self.db.shape)
            self.b = self.b - learning_rate * self.db
    
    def norm_weight(self) -> float:
        return np.sum(np.square(self.W))
    
# the activation function:

class Activation(_ActivationDense):

    def __init__(self,activation : str,n_l=None) -> None:
        super().__init__(activation,n_l)
    
    def linear_activation_forward(self,A_prev:array) -> array:
        self.A = self.activation(self.A)
        return self.A

    def linear_activation_backward(self,A_prev:array,lambd:float)->array:
        dA_prev = np.multiply(self.dA,self.backward_activation(self.Z))
        return dA_prev
    
# the dropout layer:

class Dropout(_Layer):
    def __init__(self,keep_prob = 1,n_l = None) -> None:
        super().__init__(n_l)
        self.keep_prob = keep_prob
        self.dA = None
        self.D = None
    
    def linear_activation_forward(self,A_prev:array) -> array:
        self.D = (np.random.rand(A_prev.shape[0],A_prev.shape[1]) < self.keep_prob).astype(int)
        A_prev = np.multiply(A_prev,self.D)
        A_prev = A_prev /self.keep_prob
        return A_prev
    
    def linear_activation_backward(self,A_prev:array,lambd:float)->array:
        dA_prev = np.multiply(self.D,self.dA)
        dA_prev = dA_prev / self.keep_prob
        return dA_prev
        



        
        