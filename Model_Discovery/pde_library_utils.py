"""
file describes a handcrafted pde library
"""
import numpy as np

def finite_differences(u, order=1, dt=None, dx=None, dy=None):
    """
    Computes the 1st order and 2nd order derivative with the Finite
    Difference method in a vectorized version
    order == 1: d/di (u)
    order == 2: d^2/di^2 (u)
    """

    if order == 1:
        """
        Vectorized Finite difference solver with respect to i
        dudi[i] = (u[i+1] - u[i-1]) / (2*di)
        To avoid ghost points for the boundary I chose the following:
        left boundary - forward:    dudt[0] = (u[i+1] - u[i]) / di
        right boudnary - backward:  dudt[-1] = (u[i] - u[i-1]) / di
        """
        if dt != None:
            # To vectorize the problem we will define:
            u_e = u[...,1:]   # east: u[t+1]
            u_w = u[...,:-1]  # west: u[t-1]
            # left and right boundary, as well as the central part:
            dudi_w = (u_e[...,0] - u[...,0]) / dt
            dudi_e = (u[...,-1] - u_w[...,-1]) / dt
            dudi_c = (u_e[...,1:] - u_w[...,:-1]) / (2*dt)

            # TODO: Check the results for all PDEs!
            if len(u.shape) == 2:
                result = np.concatenate([dudi_w.reshape(-1,1), dudi_c, dudi_e.reshape(-1,1)], axis=1)
            elif len(u.shape) == 3:
                result = np.concatenate([dudi_w.reshape(dudi_w.shape[0],dudi_w.shape[1],1), 
                                         dudi_c, 
                                         dudi_e.reshape(dudi_e.shape[0], dudi_e.shape[1], 1)], axis=2)
            else:
                raise ValueError("The input u given to the function has an incorrect shape")

        elif dx != None:
            u_n = u[1:,...]     # north: u[x+1]
            u_s = u[:-1,...]    # south: u[x-1]
            # left and right boundary, as well as the central part:
            dudi_s = (u_n[0,...] - u[0,...]) / dx
            dudi_n = (u[-1,...] - u_s[-1,...]) / dx
            dudi_c = (u_n[1:,...] - u_s[:-1,...]) / (2*dx)
            # result concatenated for axis 0
            if len(u.shape) == 2:
                result = np.concatenate([dudi_s.reshape(1,-1), dudi_c, dudi_n.reshape(1,-1)], axis=0)
            elif len(u.shape) == 3:
                result = np.concatenate([dudi_s.reshape(1, dudi_s.shape[0], dudi_s.shape[1]), 
                                         dudi_c, 
                                         dudi_n.reshape(1, dudi_n.shape[0], dudi_n.shape[1])], axis=0)
            else:
                raise ValueError("The input u given to the function has an incorrect shape")

        elif dy != None:
            # case 3 only valid for pde_3 not for pde_1 and pde_2
            assert len(u.shape) == 3, "FD with respect to y can only be computed for PDE Problem 3"
            u_f = u[:,1:,:]     # forward: u[y+1]
            u_b = u[:,:-1,:]    # backward: u[y-1]
            # top and bottom boundary of the xy plane, as well as the central part:
            dudi_b = (u_f[:,0,:] - u[:,0,:]) / dy
            dudi_t = (u[:,-1,:] - u_b[:,-1,:]) / dy
            dudi_c = (u_f[:,1:,:] - u_b[:,:-1,:]) / (2*dy)

            if len(u.shape) == 3:
                result = np.concatenate([dudi_b.reshape(dudi_b.shape[0], 1, dudi_b.shape[1]), 
                                        dudi_c, 
                                        dudi_t.reshape(dudi_t.shape[0],1, dudi_t.shape[1])], axis=1)
            else:
                raise ValueError("The input u given to the function has an incorrect shape")
            
    else:
        raise ValueError('Only first and second order Finite Difference schemes are valid')
    
    return result


def get_library_terms(order, arg_arr, problem_dim=1, mixed=True):
    """
    function returns a handcrafted library to avoid redundancy there
    would have been better methods as done in pde_find.py with combinations
    - order:        defines the order of nonlinearity and derivatives that 
                    are included
    - arg_arr:      tuple of predefined derivatives 
    - problem_dim:  defines if the problem has another coupled equation or
                    not (valid is the dimension 1 and 2)
    - return:       list of derivatives and corresponding library
    """

    assert order == 2 or order == 3, "Order can only be 2 or 3"
    assert problem_dim > 0 and problem_dim < 3, \
        "This library only works for a maximum of two coupled equations"

    if order == 2:

        if problem_dim == 1:
            # unpack params
            (u, u_x, u_xx) = arg_arr

        elif problem_dim == 2:
            #unpack params
            if mixed == True:
                # mixed derivatives are allowed
                (u, u_x, u_xx, v, v_x, v_xx, \
                 u_y, u_yy, v_y, v_yy, u_xy, v_xy) = arg_arr
            else:
                (u, u_x, u_xx, v, v_x, v_xx, \
                 u_y, u_yy, v_y, v_yy) = arg_arr
        
        # General setup of terms that are included anyway
        terms = [
            np.ones_like(u), u, u**2, 
            u_x, u_xx, u*u_x, u*u_xx
            ]
        
        library = [
            '1', 'u', 'u*u',
            'u_x', 'u_xx', 'u*u_x', 'u*u_xx'
            ]
        
        if problem_dim == 2:
            # additional terms if there are 2 coupled equations
            if mixed == True:
                # include mixed derivative terms
                additional_terms = [
                    # linear v terms
                    v, v**2, u**3, v**3,
                    # combinations of u and v terms
                    u*v, v*u**2, u*v**2,
                    # lin and nonlin combinations u derivatives
                    v*u_x, v*u_xx, u_y, u_yy, u*u_y, v*u_y,
                    u*u_yy, v*u_yy,
                    # mixed u derivatives
                    u_xy, u*u_xy, v*u_xy, 
                    # v spatial derivativees
                    v_x, v_xx, v_y, v_yy,
                    # lin and non lin combinations v derivatives
                    u*v_x, v*v_x, u*v_xx, v*v_xx, u*v_y, v*v_y,
                    u*v_yy, v*v_yy,
                    # mixed v derivatives
                    v_xy, u*v_xy, v*v_xy
                    ]
                
                additional_library = [
                    # linear v terms
                    'v', 'v*v', 'u*u*u', 'v*v*v',
                    # combinations of u and v terms
                    'u*v', 'u*u*v', 'u*v*v',
                    # lin and nonlin combinations u derivatives
                    'v*u_x', 'v*u_xx', 'u_y', 'u_yy', 'u*u_y', 'v*u_y',
                    'u*u_yy', 'v*u_yy',
                    # mixed u derivatives
                    'u_xy', 'u*u_xy', 'v*u_xy',
                    # v spatial derivativees
                    'v_x', 'v_xx', 'v_y', 'v_yy',
                    # lin and non lin combinations v derivatives
                    'u*v_x', 'v*v_x', 'u*v_xx', 'v*v_xx', 'u*v_y', 'v*v_y',
                    'u*v_yy', 'v*v_yy',
                    # mixed v derivatives
                    'v_xy', 'u*u_xy', 'v*u_xy'
                    ]
            else:
                # neglect mixed derivative terms
                additional_terms = [
                    # linear v terms
                    v, v**2, u**3, v**3,
                    # combinations of u and v terms
                    u*v, v*u**2, u*v**2,
                    # lin and nonlin combinations u derivatives
                    v*u_x, v*u_xx, u_y, u_yy, u*u_y, v*u_y,
                    u*u_yy, v*u_yy,
                    # v spatial derivativees
                    v_x, v_xx, v_y, v_yy,
                    # lin and non lin combinations v derivatives
                    u*v_x, v*v_x, u*v_xx, v*v_xx, u*v_y, v*v_y,
                    u*v_yy, v*v_yy,
                    ]
                
                additional_library = [
                    # linear v terms
                    'v', 'v*v', 'u*u*u', 'v*v*v',
                    # combinations of u and v terms
                    'u*v', 'u*u*v', 'u*v*v',
                    # lin and nonlin combinations u derivatives
                    'v*u_x', 'v*u_xx', 'u_y', 'u_yy', 'u*u_y', 'v*u_y',
                    'u*u_yy', 'v*u_yy',
                    # v spatial derivativees
                    'v_x', 'v_xx', 'v_y', 'v_yy',
                    # lin and non lin combinations v derivatives
                    'u*v_x', 'v*v_x', 'u*v_xx', 'v*v_xx', 'u*v_y', 'v*v_y',
                    'u*v_yy', 'v*v_yy',
                    ]
                
            # final library for a second order derivative problem 
            terms = terms + additional_terms
            library = library + additional_library
        
    elif order == 3:

        if problem_dim == 1:
            # unpack params
            (u, u_x, u_xx, u_xxx) = arg_arr

        elif problem_dim == 2:
            # unpack params
            if mixed == True:
                # include mixed derivatives
                (u, u_x, u_xx, u_xxx, v, v_x, v_xx, v_xxx, \
                u_y, u_yy, u_yyy, v_y, v_yy, v_yyy, \
                u_xy, u_xxy, u_xyy, v_xy, v_xxy, v_xyy) = arg_arr
            else:
                # exclude mixed derivatives
                (u, u_x, u_xx, u_xxx, v, v_x, v_xx, v_xxx, \
                u_y, u_yy, u_yyy, v_y, v_yy, v_yyy) = arg_arr

        # General setup of terms that are included anyway
        terms = [
            np.ones_like(u), 
            u, u**2, u**3, 
            u_x, u_xx, u_xxx, 
            u*u_x, u*u*u_x,
            u*u_xx, u*u*u_xx,
            u*u_xxx, u*u*u_xxx
            ]
        
        library = [
            '1', 'u', 'u*u', 'u*u*u', 
            'u_x', 'u_xx', 'u_xxx',
            'u*u_x', 'u*u*u_x', 
            'u*u_xx', 'u*u*u_xx',
            'u*u_xxx', 'u*u*u_xxx'
            ]
        
        if problem_dim == 2:
            # additional terms if therea are 2 coupled eqs
            if mixed:
                # mixed derivatives are included
                additional_terms = [
                    # linear v terms
                    v, v**2, v**3,
                    # combinations of u and v terms
                    u*v, v*u**2, u*v**2,
                    # spatial derivative of u 
                    u_y, u_yy, u_yyy,
                    # lin and nonlin combinations u derivatives
                    v*u_x, v*v*u_x, v*u_xx, v*v*u_xx, u*v*u_x, 
                    u*v*u_xx, v*u_xxx, v*v*u_xxx, u*v*u_xxx, 
                    u*u_y, u*u*u_y, u*v*u_y, v*u_y, v*v*u_y,
                    u*u_yy, u*u*u_yy, u*v*u_yy, v*u_yy, v*v*u_yy,
                    u*u_yyy, u*u*u_yyy,
                    v*u_yyy, v*v*u_yyy, u*v*u_yyy,
                    # mixed u derivatives
                    u_xy, u_xxy, u_xyy,
                    u*u_xy, u*u*u_xy, v*u_xy, 
                    u*v*u_xy, v*v*u_xy,
                    u*u_xxy, u*u*u_xxy, v*u_xxy, 
                    u*v*u_xxy, v*v*u_xxy,
                    u*u_xyy, u*u*u_xyy, v*u_xyy, 
                    u*v*u_xyy, v*v*u_xyy,
                    # v spatial derivativees
                    v_x, v_xx, v_xxx,
                    v_y, v_yy, v_yyy,
                    # lin and non lin combinations v derivatives
                    u*v_x, u*u*v_x, v*v_x,
                    v*v*v_x, u*v*v_x,
                    u*v_xx, u*u*v_xx, v*v_xx,
                    v*v*v_xx, u*v*v_xx,
                    u*v_xxx, u*u*v_xxx, v*v_xxx,
                    v*v*v_xxx, u*v*v_xxx,
                    u*v_y, u*u*v_y, v*v_y,
                    v*v*v_y, u*v*v_y,
                    u*v_yy, v_yy*u**2, v*v_yy,
                    v_yy*v**2, u*v*v_yy,
                    u*v_yyy, v_yyy*u**2, v*v_yyy,
                    v_yyy*v**2, u*v*v_yyy,
                    # mixed v derivatives
                    v_xy, v_xxy, v_xyy,
                    u*v_xy, v_xy*u**2, v*v_xy,
                    v_xy*v**2, u*v*v_xy,
                    u*v_xxy, v_xxy*u**2, v*v_xxy,
                    v_xxy*v**2, u*v*v_xxy,
                    u*v_xyy, v_xyy*u**2, v*v_xyy,
                    v_xyy*v**2, u*v*v_xyy
                    ]
                additional_library = [
                    # linear v terms
                    'v', 'v*v', 'v*v*v',
                    # combinations of u and v terms
                    'u*v', 'u*u*v', 'u*v*v',
                    # spatial derivative of u 
                    'u_y', 'u_yy', 'u_yyy',
                    # lin and nonlin combinations u derivatives
                    'v*u_x', 'v*v*u_x', 'v*u_xx', 'v*v*u_xx', 'u*v*u_x', 
                    'u*v*u_xx', 'v*u_xxx', 'v*v*u_xxx', 'u*v*u_xxx',
                    'u*u_y', 'u*u*u_y', 'u*v*u_y',
                    'v*u_y', 'v*v*u_y', 
                    'u*u_yy', 'u*u*u_yy', 'u*v*u_yy',
                    'v*u_yy', 'v*v*u_yy',
                    'u*u_yyy', 'u*u*u_yyy',
                    'v*u_yyy', 'v*v*u_yyy', 'u*v*u_yyy',
                    # mixed u derivatives
                    'u_xy', 'u_xxy', 'u_xyy',
                    'u*u_xy', 'u*u*u_xy', 'v*u_xy', 
                    'u*v*u_xy', 'v*v*u_xy',
                    'u*u_xxy', 'u*u*u_xxy', 'v*u_xxy', 
                    'u*v*u_xxy', 'v*v*u_xxy',
                    'u*u_xyy', 'u*u*u_xyy', 'v*u_xyy', 
                    'u*v*u_xyy', 'v*v*u_xyy',
                    # v spatial derivativees
                    'v_x', 'v_xx', 'v_xxx',
                    'v_y', 'v_yy', 'v_yyy',
                    # lin and non lin combinations v derivatives
                    'u*v_x', 'u*u*v_x', 'v*v_x', 
                    'v*v*v_x', 'u*v*v_x',
                    'u*v_xx', 'u*u*v_xx', 'v*v_xx',
                    'v*v*v_xx', 'u*v*v_xx',
                    'u*v_xxx', 'u*u*v_xxx', 'v*v_xxx',
                    'v*v*v_xxx', 'u*v*v_xxx',
                    'u*v_y', 'u*u*v_y', 'v*v_y',
                    'v*v*v_y', 'u*v*v_y',
                    'u*v_yy', 'u*u*v_yy', 'v*v_yy',
                    'v*v*v_yy', 'u*v*v_yy',
                    'u*v_yyy', 'u*u*v_yyy', 'v*v_yyy',
                    'v*v*v_yyy', 'u*v*v_yyy',
                    # mixed u derivatives
                    'v_xy', 'v_xxy', 'v_xyy',
                    'u*u_xy', 'u*u*v_xy', 'v*u_xy',
                    'v*v*u_xy', 'u*v*u_xy',
                    'u*v_xxy', 'u*u*v_xxy', 'v*v_xxy',
                    'v*v*v_xxy', 'u*v*v_xxy',
                    'u*v_xyy', 'u*u*v_xyy', 'v*v_xyy',
                    'v*v*v_xyy', 'u*v*v_xyy'
                    ]
            else:
                # mixed derivatives are excluded
                additional_terms = [
                    # linear v terms
                    v, v**2, v**3,
                    # combinations of u and v terms
                    u*v, v*u**2, u*v**2,
                    # spatial derivative of u 
                    u_y, u_yy, u_yyy,
                    # lin and nonlin combinations u derivatives
                    v*u_x, v*v*u_x, v*u_xx, v*v*u_xx, u*v*u_x, 
                    u*v*u_xx, v*u_xxx, v*v*u_xxx, u*v*u_xxx, 
                    u*u_y, u*u*u_y, u*v*u_y, v*u_y, v*v*u_y,
                    u*u_yy, u*u*u_yy, u*v*u_yy, v*u_yy, v*v*u_yy,
                    u*u_yyy, u*u*u_yyy,
                    v*u_yyy, v*v*u_yyy, u*v*u_yyy,
                    # v spatial derivativees
                    v_x, v_xx, v_xxx,
                    v_y, v_yy, v_yyy,
                    # lin and non lin combinations v derivatives
                    u*v_x, u*u*v_x, v*v_x,
                    v*v*v_x, u*v*v_x,
                    u*v_xx, u*u*v_xx, v*v_xx,
                    v*v*v_xx, u*v*v_xx,
                    u*v_xxx, u*u*v_xxx, v*v_xxx,
                    v*v*v_xxx, u*v*v_xxx,
                    u*v_y, u*u*v_y, v*v_y,
                    v*v*v_y, u*v*v_y,
                    u*v_yy, v_yy*u**2, v*v_yy,
                    v_yy*v**2, u*v*v_yy,
                    u*v_yyy, v_yyy*u**2, v*v_yyy,
                    v_yyy*v**2, u*v*v_yyy
                    ]
                additional_library = [
                    # linear v terms
                    'v', 'v*v', 'v*v*v',
                    # combinations of u and v terms
                    'u*v', 'u*u*v', 'u*v*v',
                    # spatial derivative of u 
                    'u_y', 'u_yy', 'u_yyy',
                    # lin and nonlin combinations u derivatives
                    'v*u_x', 'v*v*u_x', 'v*u_xx', 'v*v*u_xx', 'u*v*u_x', 
                    'u*v*u_xx', 'v*u_xxx', 'v*v*u_xxx', 'u*v*u_xxx',
                    'u*u_y', 'u*u*u_y', 'u*v*u_y',
                    'v*u_y', 'v*v*u_y', 
                    'u*u_yy', 'u*u*u_yy', 'u*v*u_yy',
                    'v*u_yy', 'v*v*u_yy',
                    'u*u_yyy', 'u*u*u_yyy',
                    'v*u_yyy', 'v*v*u_yyy', 'u*v*u_yyy',
                    # v spatial derivativees
                    'v_x', 'v_xx', 'v_xxx',
                    'v_y', 'v_yy', 'v_yyy',
                    # lin and non lin combinations v derivatives
                    'u*v_x', 'u*u*v_x', 'v*v_x', 
                    'v*v*v_x', 'u*v*v_x',
                    'u*v_xx', 'u*u*v_xx', 'v*v_xx',
                    'v*v*v_xx', 'u*v*v_xx',
                    'u*v_xxx', 'u*u*v_xxx', 'v*v_xxx',
                    'v*v*v_xxx', 'u*v*v_xxx',
                    'u*v_y', 'u*u*v_y', 'v*v_y',
                    'v*v*v_y', 'u*v*v_y',
                    'u*v_yy', 'u*u*v_yy', 'v*v_yy',
                    'v*v*v_yy', 'u*v*v_yy',
                    'u*v_yyy', 'u*u*v_yyy', 'v*v_yyy',
                    'v*v*v_yyy', 'u*v*v_yyy'
                    ]
                
            # combine additional terms
            terms = terms + additional_terms
            library = library + additional_library

    return terms, library

def initialize_derivatives(di_args, u, v=None, problem_dim=1, deriv=False):
    """
    - di_args:      (float - dx, dy, dt) includes the time and spacial stepping
    - u:            first solution tensor
    - v:            second solution tensor if problem_dim == 2
    - problem_dim:  (int 1 or 2) number of coupled equations
    - deriv:        (bool) gives back necessary derivatives for evaluation
    """
    if problem_dim == 1:
        (dx, dt) = di_args

    if problem_dim == 2:
        (dx, dy, dt) = di_args
    
    # Time Derivative t
    u_t = finite_differences(u, dt=dt)

    # Space Derivatives x
    u_x = finite_differences(u, dx=dx)
    u_xx = finite_differences(u_x, dx=dx) # 2nd order
    u_xxx = finite_differences(u_xx, dx=dx) # 3rd order
    # 4th order spatial derivative only for plotting and comparison
    u_xxxx= finite_differences(u_xxx, dx=dx) # two times 2nd order

    
    if problem_dim == 2:
        # Time Derivative v
        v_t = finite_differences(v, dt=dt)
        # Space Derivatives y
        u_y = finite_differences(u, dy=dy)
        u_yy = finite_differences(u_y, dy=dy) # 2nd order
        u_yyy = finite_differences(u_yy, dy=dy) # 3rd order
        # Mixed Derivatives u
        u_xy = finite_differences(u_x, dy=dy)
        u_xxy = finite_differences(u_xx, dy=dy)
        u_xyy = finite_differences(u_xy, dy=dy) # 2nd order

        # Space Derivatives v
        v_x = finite_differences(v, dx=dx)
        v_xx = finite_differences(v_x, dx=dx) # 2nd
        v_xxx = finite_differences(v_xx, dx=dx) # 3rd
        v_y = finite_differences(v, dy=dy)
        v_yy = finite_differences(v_y, dy=dy) # 2nd order
        v_yyy = finite_differences(v_yy, dy=dy) # 3rd order

        # Mixed Derivatives v
        v_xy = finite_differences(v_x, dy=dy)
        v_xxy = finite_differences(v_xx, dy=dy) # 2nd order
        v_xyy = finite_differences(v_xy, dy=dy) # 2nd order

        args = (u_x, u_xx, u_xxx, v_x, v_xx, v_xxx, \
                u_y, u_yy, u_yyy, v_y, v_yy, v_yyy, \
                u_xy, u_xxy, u_xyy, v_xy, v_xxy, v_xyy, \
                v_t)
        
        arg_list = ('u_x', 'u_xx', 'u_xxx', 'v_x', 'v_xx', 'v_xxx', \
                    'u_y', 'u_yy', 'u_yyy', 'v_y', 'v_yy', 'v_yyy', \
                    'u_xy', 'u_xxy', 'u_xyy', 'v_xy', 'v_xxy', 'v_xyy',
                    'v_t')
        
    if problem_dim == 1:
        args = (u_x, u_xx, u_xxx)
        arg_list = ('u_x', 'u_xx', 'u_xxx')
    
    if deriv:
        return (u_t, u_x, u_xx, u_xxx, u_xxxx)
    
    else:
        return args, arg_list