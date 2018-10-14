import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

def project_onto_subspace(A, y):
    """
    project y onto subspace defined by the columns of A.
    
    :param: A -- the (2 x 1) matrix defining the subspace to project onto.
    :param: y -- the data matrix of observations in the original 2D space.
    
    return: 
        0: co-ordinates of projected point in original space
        1: co-ordinates of projected point in subspace defined by A
    """
    assert A.ndim == 2, "A must have 2 dimensions"
    (A.shape[1] > 1) and warn("projection function expecting 1 column of A. Found {:d}.".format(
        A.shape[1]))
    
    # taking care of dimensional consistency where reqd
    if y.ndim == 2:
        if y.shape[1] == 2 and y.shape[0] > 1:
            y = y.T   # long => wide form
        elif y.shape[0] <=2:
            pass      # ok already
        else:
            raise Exception("y should be a vector of length 2 or matrix 2 x n. Got {0}".format(y.shape))
        
    # actual projection
    w = np.linalg.solve(A.T @ A, A.T @ y)
    return w.T @ A.T, w.ravel()

    
def plot_projection(X, projX, ax=None):
    """
    plot the projections of the data.
    
    :param: X -- the original (n x 2) data matrix
    :param: projX -- the (n x 2) matrix of projected data co-ordinates.
    :param: ax -- [OPTIONAL] - the axis handle on which to plot.
    """
    assert X.shape == projX.shape, "X and projX have different shapes: {0}, {1}".format(X.shape, projX.shape)
    assert X.shape[1] == 2, "X, projX should have 2 columns, {:d} found".format(X.shape[1])
    
    ax = plt.gca() if ax == None else ax
    
    ax.scatter(*X.T)
    ax.scatter(*projX.T)
    for i in range(X.shape[0]):
        ax.plot([X[i,0], projX[i,0]], [X[i,1], projX[i,1]], color="grey", linewidth=0.6, zorder=0)
        
    # calculate the slope in a slightly hacky way / plot the subspace as a line.
    regr_slope = np.divide(*np.diff(projX[:2,:], 1, 0)[0][::-1])   # ? infty issues seem ok.
    
    limsx, limsy = ax.get_xlim(),ax.get_ylim()
    xrng = np.diff(limsx)[0]
    ax.arrow(limsx[0],limsx[0]*regr_slope, xrng, xrng*regr_slope, width=0.2, head_width=.0, 
              head_length=.0, color="red", zorder=0)
    
    # deal with axis limits => this is important.
    ax.set_xlim(limsx); ax.set_ylim(limsy)
    ax.set_aspect('equal', 'datalim')   # ciritical for visualisation to work. Aspect ratio must be correct.
    

def plot_abline(slope, intercept, *args, **kwargs):
    """
    Plot a line from slope and intercept.
    If desired, an axis handle can be specified.
    """
    ax = plt.gca() if "ax" not in kwargs else kwargs.pop("ax")
    limsx, limsy = ax.get_xlim(),ax.get_ylim()
    x_vals = np.array(ax.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, *args, **kwargs)
    ax.set_xlim(limsx); ax.set_ylim(limsy)
    
    
def generate_data(n):
    n1 = n2 = n // 2
    n1 = n1 + 1 if n % 2 == 1 else n1
    _tmp = np.concatenate([np.array([0., 13.]) + np.random.randn(n1,2)@np.array([[3.2, 0], [2.0, 1.9]]),
                       np.array([14., 17.]) + np.random.randn(n2,2)@np.array([[4., 0], [0.05, 1.3]])])
    x, y = _tmp[:,0:1], _tmp[:,1:2]
    return x, y