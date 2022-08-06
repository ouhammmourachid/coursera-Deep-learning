
# libraries that we will need are
from Layers import EPSILON, Dropout
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from numpy import float32

# definition of the Network class

class Network:
    def __init__(self,layers:list) -> None:
        if not isinstance(layers[0],InputLayer):
            layers = [InputLayer()] + layers
        self.layers = layers
        self.n = len(self.layers)
        
    
    # intialize weights function :

    def initialize_weights(self):
        for i in range(1,self.n):
            self.layers[i].initialize_weights(self.layers[i-1].n_l)



    def model_forward(self,X:array,predict =  False) -> None :
        A = X
        for layer in self.layers:
            if predict and  isinstance(layer,Dropout) :
                continue
            A = layer.linear_activation_forward(A)
    
    # cost function with the term of regularisation in case if lambd != 0

    def cost(self,Y,lambd:float) -> float:
        
        # we simplie retrive AL from the last layers and Y from the input layer:

        AL = self.layers[-1].A
        m = AL.shape[1]
        Y = self.layers[0].flatten_y(Y)

        # here we compute the cost function without any regularisation terme:

        cost = np.squeeze(-1/m*np.sum(np.multiply(Y,np.log(AL + EPSILON))+np.multiply(1-Y,np.log(1-AL +EPSILON))))

        # here i add regularization terme if the hyperparamters lambd is != 0

        if lambd != 0:
            cost += lambd/(2*Y.shape[1])*self.cost_regularisation()
        return cost
    
    # to compute the terme of regularization in cost function

    def cost_regularisation(self) -> float:
        cost_regularisation = 0
        for layer in  self.layers:
            cost_regularisation += layer.norm_weight()
        return cost_regularisation
    
    # the function that performe backpropagation

    def model_backward(self,X:array,Y:array,lambd:float) -> None:

        AL = self.layers[-1].A
        Y = self.layers[0].flatten_y(Y)
        self.layers[-1].dA = np.divide(1-Y,1-AL) - np.divide(Y,AL)
        for i in range(1,self.n):
            A_prev = self.layers[-i-1].A
            self.layers[-i-1].dA = self.layers[-i].linear_activation_backward(A_prev,lambd)
    
    # update the weights of each layer

    def model_update_weight(self,learning_rate:float)->None:
        for layer in self.layers:
            layer.update(learning_rate)
    
    def fit(self,X:array,Y:array,epochs:int,learning_rate = 0.01,print_cost = True,lambd = 0) -> list:
        
        # set the th number of units in the input layer to fit the size of x.shape[0]

        if not isinstance(self.layers[0],Flatten) :
            self.layers[0].n_l = X.shape[0]

         # intilize the wweight of the layers :

        self.initialize_weights()

        # create a list of costs and acurreces to stores the result of our network after each mini batch

        costs = list()
        accuracies = list()

        for epoch in tqdm(range(epochs)):

            
            self.model_forward(X)
            self.model_backward(X,Y,lambd)
            self.model_update_weight(learning_rate)


            cost = self.cost(Y,lambd)
            accuracy = accuracy_score(Y.flatten(),self.predict(X).flatten())
            if print_cost:
                if epoch % 100 == 0 or epoch == epochs -1:
                    print(f'Epoch : {epoch} cost --> {cost} / accur ---> {accuracy}')
            costs.append(cost)
            accuracies.append(accuracy)
        return costs,accuracies
    
    def predict(self,X_test:array)->array:
        self.model_forward(X_test)
        return self.layers[-1].A > 0.5
            
