import numpy as np

def conjugate_grad_solve(fun, mask, b, x, tol=1e-9, niter=50):
    # Copy the initial guess x into a new vector which we will modify
    x_solve = np.copy(x)

    # Calculate the residual with the initial guess
    r = b - fun(x, mask)

    # Calculate the length squared of the residual
    rsquared = np.sum(r * r)

    # Copy the residual into a new vector (representing the basis vector p_i)
    p = np.copy(r)

    # This flag to see if it worked
    converged = False

    # Loop through once for each dimension
    for i in range(niter):
        # Calculate the coefficient alpha for the new solution
        Ap = fun(p, mask)
        alpha = rsquared / np.sum(p * Ap)

        # Update the solution vector using alpha and p
        x_solve = x_solve + alpha * p

        # Calculate the new residuals and length squared
        r = r - alpha * Ap
        rsquared_new = np.sum(r * r)
        
        # Print something to show iteration progress
        print("residuals =", rsquared_new,"iteration", i+1, "out of", niter)

        # If the residual is smaller than the tolerance, break and return
        if np.sqrt(rsquared_new) < tol:
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