import numpy as np
from scipy.optimize import BFGS, SR1
import trustregion
from scipy.optimize import OptimizeResult
from bpllib._utils import _to_array

def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75:
        Delta *= 2.0

    return Delta, ratio

def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """Check termination condition"""
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None
    
def print_iteration(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality,radius,Bcond):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.2e}".format(cost_reduction)

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    print("{0:^15}{1:^15}{2:^15.4e}{3}{4}{5:^15.2e}{6:^15.2e}{7:^15.2e}"
          .format(iteration, nfev, cost, cost_reduction,
                  step_norm, optimality,radius,Bcond))


def print_header():
    print("\n{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}{6:^15}{7:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality", "TR-Radius", "B Cond"))

def trustregion_box(
    fun,
    grad,
    reg_grad,
    x0,
    verbose=False,
    initial_radius=1.0,
    threshold_radius=0.1,
    max_radius=100,
    max_nfev=2000,
    xtol=1e-7,
    ftol=1e-7,
    radius_tol=1e-7,
    reg_gradient_only=False
    ):
    '''
    Trust Region Algorithm with positivity constraints
    '''
    
    # initialization
    x0 = _to_array(x0,'x0')
    x = x0
    radius = initial_radius
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    nfev = 1
    njev = 0
    n_reg_jev = 0
    B = BFGS(init_scale=1.0)
    B.initialize(len(x),'hess')
    njev += 1
    g = grad(x)
    print_header()
    print_iteration(iteration,nfev,fun(x),0,0,np.linalg.norm(g),radius,np.linalg.cond(B.get_matrix()))
    while True:
        sl = -x - 1e-9
        su = 1e9 - x
        fx = fun(x)
        s = trustregion.solve(g,B.get_matrix(),radius,sl=sl,su=su)
        s_norm = np.linalg.norm(s)
        x_ = x + s
        pred = -np.dot(g,s)-0.5*np.dot(s,B.dot(s))
        ared = fx-fun(x_)
        radius, ratio = update_tr_radius(radius,ared,pred,s_norm)
        if radius>max_radius:
            radius = max_radius
        
        # Checking termination
        termination_status = check_termination(ared,fx,s_norm,np.linalg.norm(x),ratio,ftol,xtol)
        
        if radius < radius_tol:
            termination_status = 3
            
        if nfev > max_nfev:
            termination_status = 5

        if termination_status is not None:
            break
        
        if ared > 0:
            x = x_
            fx = fun(x)
            nfev += 1
            if radius >= threshold_radius:
                if np.dot(s, grad(x_)-g) > 1e-9:
                    B.update(s,grad(x_)-g)
                njev +=1
                g = grad(x_) 
                # print(f'g:{g}')
            else:
                if np.dot(s, reg_grad(x_)-g) > 1e-9:
                    B.update(s,reg_grad(x_)-g)
                n_reg_jev +=1
                g = reg_grad(x_)
        else:
            #B.initialize(len(x.data),'hess')
            s = 0
            ared = 0
            radius *= 0.5

        iteration += 1
        print_iteration(iteration,nfev,fx,ared,s_norm,np.linalg.norm(g),radius,np.linalg.cond(B.get_matrix()))
        
    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x, fun=fx, jac=g, optimality=np.linalg.norm(g), nfev=nfev, njev=njev, n_reg_jev=n_reg_jev,nit=iteration, status=termination_status)
        