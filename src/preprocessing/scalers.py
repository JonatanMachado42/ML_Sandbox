import numpy as np 

class MinMaxScaler : 
    def __init__(self) :
        self.min = None 
        self.max = None 
    
    def fit(self,X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self,X): 
        denom = self.max - self.min 
        denom[ denom == 0] = 1 # Evita divisao por 0 
        return (X - self.min) / denom
        