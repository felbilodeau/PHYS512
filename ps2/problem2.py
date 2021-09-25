import numpy as np

def integrate_adaptive(fun, a, b, tol, extra = None):

    # Here we either create a dictionary or
    # we take the extra one and pass it along
    if extra == None:
        values = {}
    else:
        values = extra

    # We are using Simpson's like we did in class
    x = np.linspace(a, b, 5)
    y = np.zeros((len(x)))

    # Here we check if we already had the value
    # This makes sure we only call the function once
    # per value of x
    for i in range(len(y)):
        if values.get(x[i]) == None:
            y[i] = fun(x[i])
            values[x[i]] = y[i]
        else:
            y[i] = values.get(x[i])
        
    # Calculate the current dx
    dx = (b - a) / (len(x) - 1)

    # Calculate a coarse and fine step
    area1 = 2*dx * (y[0] + 4*y[2] + y[4]) / 3 # coarse

    area2 = dx * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4]) / 3 # fine

    # Calcuoate the error based on the two steps
    err = np.abs(area1 - area2)

    # If the error is smaller than our tolerance,
    # we return the fine step.
    # If it is not, we half the step size
    if err < tol:
        return area2

    else:
        # Calculate the midpoint
        xmid = (a + b) / 2

        # Recalculate the left and right halves
        left = integrate_adaptive(fun, a, xmid, tol/2, values)
        right = integrate_adaptive(fun, xmid, b, tol/2, values)
        return left + right

if __name__ == "__main__":
    
    from integrate_adaptive import integrate_adaptive as integrate_adaptive_class

    # Define the variable for the number of function calls
    function_calls = 0

    # This is our test function
    def fun(x):
        # Here we increment the function_calls variable globally
        global function_calls
        function_calls += 1

        return np.sin(x)

    # Set our parameters
    a = 0
    b = np.pi
    tol = 1e-13
    true_answer = np.cos(a) - np.cos(b)

    # Do the integration with my function
    integral_mine = integrate_adaptive(fun, a, b, tol)
    err_mine = np.abs(true_answer - integral_mine)
    print("my integral =", integral_mine)
    print("my function calls =", function_calls)
    print("my error =", err_mine)

    # Print an empty line and reset function_calls
    print()
    function_calls = 0

    # Do the integration with the function from github
    integral_class = integrate_adaptive_class(fun, a, b, tol)
    err_class = np.abs(true_answer - integral_class)
    print("class integral =", integral_class)
    print("class function calls =", function_calls)
    print("class error =", err_class)

    # So if you run this, you should see that my function takes
    # 4203 function calls to do this, while the class function takes
    # 10015 function calls to do the same thing, so we've reduced our
    # number of function calls by a factor of around 2.4