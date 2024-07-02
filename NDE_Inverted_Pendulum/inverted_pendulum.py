import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Dopri5
from ode import f, vector_field
from neural_network import FCN
import equinox as eqx
from util import timeit, plot_train_development
import optax

class Inverted_Pendulum:
    "Defines the Simulation and Trainer for the Inverted Pendulum Problem"

    def __init__(self, t0, tN, N, dt, x0, args):
        "Arguments necessary to initialize the simulation and NDE"

        # Initialize the solver
        self.t0 = t0 # Initial simulation
        self.tN = tN # Final simulation time
        self.N = N # Number of evaluations
        self.dt = dt # Timestep
        self.x0 = x0 # (x, d_x, theta, d_theta)
        self.x0_arr = jnp.asarray(self.x0)
        self.args = args # (m, g, l, M)
        self.t = jnp.linspace(self.t0, self.tN, self.N)

        self.solver_used = "RK4" # Relevant setup for diffrax

        # Initialize model
        in_features = 1 # t, x, d_x, theta, d_theta
        hidden_features = 32
        out_features = 1 # force F
        key = jr.key(0)
        self.model = FCN(in_features, hidden_features, out_features, key)

        self.penalty = 0.1 # penalizes the rot velocity in the loss function

        # Setup for Training:
        self.lossvals = []

        self.optimiser = optax.adam(learning_rate=1e-2)
        self.opt_state = self.optimiser.init(eqx.filter(self.model, eqx.is_array))
        # opt_state = optimiser.init(eqx.filter(fargs, eqx.is_array))
        

    def diffrax_solver(self, model=None, nde=False):
        term = ODETerm(vector_field) # ADD F manually in the case of nde!
        if self.solver_used == "RK4":
            solver = Dopri5() # 4th Order Runge Kutta solver
        else:
            solver = Tsit5() # Recommended solver
        saveat = SaveAt(ts=jnp.linspace(self.t0, self.tN, self.N))

        args = (model, *self.args, nde)

        sol = diffeqsolve(term, solver, self.t0, self.tN, self.dt, self.x0, args=args, saveat=saveat) # TODO: Try to add nde here!
        return sol
    
    def ode_solver(self, model=None, debug=False, nde=False):
        "Generic 4th order explicit RK ODE solver. f(x, *fargs) is a function which computes the RHS of the ODE."

        def single_step(x, i):
            t = self.t0 + (i+1)*self.dt

            k1 = f(x, t, model, *self.args, nde=nde)
            k2 = f(x+self.dt*k1/2, t, model, *self.args, nde=nde)
            k3 = f(x+self.dt*k2/2, t, model, *self.args, nde=nde)
            k4 = f(x+self.dt*k3, t, model, *self.args, nde=nde)
            x = x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

            if debug:
                jax.debug.print("t = {t}, x = {x}", t=t, x=x)

            return x, x
        
        # jax.lax.scan returns a carry argument and a tensor X
        _, X = jax.lax.scan(single_step, self.x0_arr, jnp.arange(0, self.N-1)) # N-1 since initial guess should be included
        # We also want to include the initial guess to the solution
        X = jnp.concatenate([self.x0_arr.reshape(1,-1), X], axis=0)
        return X
    
    @timeit
    def test_method(self, solver):
        "This function returns the solution and the time it takes to run a defined solver"
        if solver == "diffrax":
            sol = self.diffrax_solver()
        else:
            sol = self.ode_solver()
        return sol

    def loss_fn(self, model, solver="RK4"):

        desired_val = 0.0
        idx = int(0.75 * self.N) 
        if solver == "RK4":
            sol = self.ode_solver(model, nde=True)

            loss_pos = jnp.mean((sol[idx:, 2] - desired_val)**2)
            loss_vel = jnp.mean((sol[idx:, 3] - desired_val)**2)

        elif solver == "diffrax":
            diffrax_res = self.diffrax_solver(model, nde=True)
            sol = diffrax_res.ys

            loss_pos = jnp.mean((sol[2][idx:] - desired_val)**2)
            loss_vel = jnp.mean((sol[3][idx:] - desired_val)**2)

        loss = loss_pos + self.penalty * loss_vel
        return loss


    def grad(self, model, solver):
        "Computes gradient of loss function with respect to model parameters"
        partial = lambda fcn_model: self.loss_fn(fcn_model, solver=solver)
        loss_value, grads = eqx.filter_value_and_grad(partial)(model)
        return loss_value, grads
    
    @eqx.filter_jit
    def step(self, model, opt_state, solver):
        "Performs one gradient descent step on model parameters"
        # Compute loss and gradients
        lossval, grads = self.grad(model, solver)
        
        # Compute parameter updates and update the optimizer state
        updates, opt_state = self.optimiser.update(grads, opt_state)
        
        # Apply the updates to the model parameters
        model = eqx.apply_updates(model, updates)
        
        return lossval, model, opt_state
    
    def train(self, epochs, solver='RK4', video=False):
        """
        Start training a Fully Connected Neural Network
        - solver: RK4 or diffrax
        RK4 is in training much faster than the RK solver of diffrax
        """
        assert solver == 'RK4' or 'diffrax', \
            "Chosen solver is not valid. Choose between 'RK4' and 'diffrax'"
        
        for i in range(epochs):
            lossval, self.model, self.opt_state = self.step(self.model, self.opt_state, solver)
            self.lossvals.append(lossval)
            
            if (i+1)%500==0 or i==0:
                print(f"[{i+1}/{epochs}] loss: {lossval}")

            if video:
                if (i+1)%10==0 or i==0:   
                    x_train = self.ode_solver(self.model, nde=True)
                    plot_train_development(x_train, self.t, i+1, self.model)
                    
                