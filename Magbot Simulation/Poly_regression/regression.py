import numpy as np

def regre(x, r, m):
    
    # Create matrix A using the Vandermonde matrix
    A = np.vander(x, m + 1, increasing=True)
    
    # Solve the least squares problem to find the coefficients
    coeff, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    
    print("Coefficients:", coeff)
    return coeff

def polynomial(x, a):
    """
    This function takes in a list of x values and a list of coefficients,
    and returns a list of y values.

    Parameters:
    x (numpy array): The x values.
    a (nmpy array): The coefficients of the polynomial.

    Returns:
    y (numpy.ndarray): The y values computed from the polynomial.
    """
    
    # Evaluate the polynomial using the coefficients
    y = np.polyval(a[::-1], x)  # Reverse coefficients for np.polyval
    return y

def cd(y_true, y_pred):
    y_mean = np.mean(y_true)

    # Calculate the total sum of squares (SS_tot)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Calculate the residual sum of squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)

    return r2

def fit_curve(x, r, m, ni):

    n = len(x)

    #create a new list of interpolated x values
    x_inter = np.linspace(x[0], x[n-1], ni)

    #find interpolated r values
    a = regre(x, r, m)
    y_inter = polynomial(x_inter, a)

    #calculate r2
    y_pred = polynomial(x, a)
    print("y_pred",y_pred)

    r2 = cd(r, y_pred)

    return x_inter, y_inter, r2

