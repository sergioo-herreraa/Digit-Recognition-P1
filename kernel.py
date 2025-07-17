import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (np.dot(X,Y.T)+c)**p
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    def squared_norms(X,Y):  
        def substract(x,y):
            return x-y
        Substract=np.vectorize(substract)
        Submat=[]
        for i in range(X.shape[0]):
            Submat.append(Substract(X[i],Y)) 
        S=np.array(Submat) 
        norms=np.linalg.norm(S,axis=2)  
        return norms**2  
    return np.exp(-gamma*squared_norms(X,Y))
    raise NotImplementedError
