import jax.numpy as jnp

def force(t):
    return 10*jnp.sin(t)

def f(x, t, model, m, g, l, M, nde=False):
    """
    First Order Inverted Pendulum ODEs

    d_x1 = x2
    d_x2 = (F - m*g*cos(x3)*sin(x3) + m*l*x4^2*sin(x3)) / (M + m - m*cos^2(x3))
    d_x3 = x4
    d_x4 = (g*sin(x3) - d_x2*cos(x3)) / l

    where: 
    d_i is the first order and dd_i corresponds to the second order derivative
    x[0] = x1 = x ... position
    x[1] = x2 = d_x ... linear velocity
    x[2] = x3 = theta ... rot angle in rad
    x[3] = x4 = d_theta ... rot velocity
    """
    F = model(jnp.array([t])) if nde else jnp.array([force(t)])
        
    d_x2 = (F - m*g*jnp.cos(x[2])*jnp.sin(x[2:3]) + \
            m*l*jnp.sin(x[2:3])*x[3:4]**2) / (M + m - m*jnp.cos(x[2:3])**2)
    
    rhs = jnp.concatenate([
        x[1:2],
        d_x2,
        x[3:4],
        (g*jnp.sin(x[2:3])/l) - (jnp.cos(x[2:3])/l) * d_x2
        ], axis=0)
    return rhs

# TODO: Resolve this if there is more time, that F actually comes as an input if nde!
# Note: I added F here!
def vector_field(t, x, args):
    x1, x2, x3, x4 = x
    model, m, g, l, M, nde = args
    F = model(jnp.array([t])) if nde else jnp.array([force(t)])
    # import ipdb; ipdb.set_trace()
    # F = force(t)
    d_x1 = x2
    d_x2 = (F[0] - m*g*jnp.cos(x3)*jnp.sin(x3) \
            + m*l*jnp.sin(x3)*(x4**2)) / (M + m - m*(jnp.cos(x3)**2))
    d_x3 = x4
    d_x4 = (g*jnp.sin(x3)/l) - (jnp.cos(x3)/l) * d_x2
    d_x = d_x1, d_x2, d_x3, d_x4
    return d_x