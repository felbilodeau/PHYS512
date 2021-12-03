import numpy as np

def conjugate_grad_solve(A, b, x, tol):
    # Copy the initial guess x into a new vector which we will modify
    x_solve = np.copy(x)

    # Calculate the residual with the initial guess
    r = b - A @ x

    # Calculate the length squared of the residual
    rsquared = r.T @ r

    # Copy the residual into a new vector (representing the basis vector p_i)
    p = np.copy(r)

    # Calculate the number of dimensions of the vector
    n = len(b)

    # This flag to see if it worked
    converged = False

    # Loop through once for each dimension
    for i in range(n):
        # Calculate the coefficient alpha for the new solution
        alpha = rsquared / (p.T @ A @ p)

        # Update the solution vector using alpha and p
        x_solve = x_solve + alpha * p

        # Calculate the new residuals and length squared
        r = r - alpha * A @ p
        rsquared_new = r.T @ r

        # If the residual is smaller than the tolerance, break and return
        if np.sqrt(r.T @ r) < tol:
            converged = True
            break
        
        # Otherwise, calculate the new basis vector and update the length of residuals squared
        p = r + (rsquared_new / rsquared) * p
        rsquared = rsquared_new

    # If we did not exceed the tolerance, print a message
    if not converged:
        print("WARNING: Did not reach desired tolerance, answer may not be accurate.")

    # Return the solution
    return x_solve