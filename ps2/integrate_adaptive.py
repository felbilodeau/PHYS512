# I copied this from github so I could compare
# the number of function calls for problem 2
# I modified the way the function is called
# so it counts every time we call the function
# on each value of x rather than the number
# of times it is called on an array

import numpy as np

def integrate_adaptive(fun,x0,x1,tol):
    #print('integrating between ',x0,x1)
    #hardwire to use simpsons
    x=np.linspace(x0,x1,5)
    
    # This is the part I changed
    y = np.zeros((len(x)))
    for i in range(len(x)):
        y[i] = fun(x[i])

    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive(fun,x0,xmid,tol/2)
        right=integrate_adaptive(fun,xmid,x1,tol/2)
        return left+right