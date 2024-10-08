# AI in Science and Engineering Projects

This Repository contains several different Projects in the Field of how to find an approximate solution for AI in Science and Engineering related tasks by using different Deep Learning Methods. These Deep Learning Architectures mainly utilize and combine Mathemtics and Physics with different Machine Learning (ML) approaches. Although, for more complex tasks, just as high turbulence flows it could be seen that most of them still fail and Finite Element or Finite Volume methods still outperform the following illustrated architectures. This said, current research in this field still show promising results. Especially for higher dimensions problem where Numerical Methods fail and the only alternative is a Monte Carlo Simulation, these approaches are able to utilize the Universal Approximation Theorem and present results.

In the following folders, there will be some problems presented, which utilize physics and math based Neural Network architecture frameworks, such as Neural Operators, more particularly Fourier Neural Operators (FNOs) or Physics informed Neural Networks (PINNs).

Moreover, this folder represents for the major part work being done for the courses AI in Science and Engineering, as well as Probabilistic Artificial Intelligence at ETH Zurich.

### Applied_Regression

This task deals with the Kaggle California Housing Challenge and presents the results for a Neural Network approach sovling a regression task. It could be seen from this example that ML approaches like Random Forest still outperform Deep Learning when it comes to predictions for this type of problem.

### Model_Discovery

One commonly used approach to tackle the problem of model discovery 
is a method called SINDy, first preseted in [SINDYc](https://www.sciencedirect.com/science/article/pii/S2405896316318298). 

An algorithm called PDE_FIND combines a symbolic regression for the 
discovery with some domain constraints and a library consisting of 
several partial derivative operators. The goal of this task is to 
achieve an inverse modelling where we have given some data of the 
solution and we want to find the underlying symbolic PDE. More 
detailed information about this task and how to tackle it can be 
found in the underlying Jupyter Notebook in the folder **Model_Discovery**.

In order to solve this task you first need to download the data from here: 
[Data](https://drive.google.com/drive/u/0/folders/1XqjAloJ7zMVqnHJqbwi_315aPqORCunm). 
Then in a next step you need to create a folder called Data and put the three npz files 
into this newly created folder inside the Model_Discovery directory.

### Inverted Pendulum

This task involves the solution of the inverted pendulum. Two different approaches in how to solve this task are given
1. **Reinforcement Learning (RL)**: The inverted pendulum problem can be solved by defining an Action and Observation space and utilizing an RL algorithm to tackle this problem. This approach can be found in the Folder **RL_Inverted_Pendulum**.
2. **Neural Differential Equation (NDE)**: Another approach is by replacing the quantity of interest for our control task with a Neural Network and redefining the ODE as an NDE. The results can be found in the folder **NDE_Inverted_Pendulum**

### FNO (Fourier Neural Operator)

This example shows that an FNO can be used to solve a coupled thermodynamics PDE problem. Though, FNO gained a lot of attention in recent years it's not a Representation equivalen Neural Operator (ReNO), meaning that due to the Shannon Nyquist frequency and bandlimited functions there will alway be an aliasing error. Thus, the results gained when trying the learn the Operator that maps an input function to a solution function in continuous space have an error. This error arrives from the fact that through a projection and lifting operation it is possible to transfer the problem from a continuous to a discrete domain and vise versa. 

It's necessary to solve the problem in a discrete setting since Neural Network solutions by using computers can only be solved in the discrete domain. There a function can be expressed by a set. Thus, we map an input set to an output set and then go back into the continuous domain. This is the basic idea behind this Neural Operator approach.

For more information, visit the following webpage [https://zongyi-li.github.io/blog/2020/fourier-pde/](https://zongyi-li.github.io/blog/2020/fourier-pde/).

### SFNO (Spherical Fourier Neural Operator)

The SFNO method shares a lot of similarities with the FNO approach and uses a specific basis transformation, more precisely the spherical harmonic transformation. One main advantage of both the FNO and SFNO is that a convolution in time domain is a multiplication in the frequency domain.

Also here, for more information visit the following webpage: [https://github.com/NVIDIA/torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

## PINNs

The key idea behind a PINN is to directly use a neural network to approximate the solution of a given problem. The nature of the problem though can be different, thus the capabilities of Pinns can be used for various tasks. This being said, also this method has some major flaws, which can be seen by looking at the next two examples presented.

First just a brief introduction about PINNs, they make use of collocation points and are essentially useful in the case of PDEs or Ordinary Differential Equations (ODEs). Since these problems incorporate either a boundary or initial condition a special loss function is required. PINNs essentially incorporate two types of Loss functions:

- Boundary Loss: The network tries to learn the inital and boundary condition
- Physics Loss: This is also called the PDE or ODE residual.

More information can be found here: [https://github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)

### PINN_PDE

In this problem PINNs were used to solve a coupled thermodynamics PDE problem. Thus the first use of PINNs is to use them in order to find an approximated solution.

### PINN_Inverse

Inverse Problems are search problems or can be seen as such. Thus, they can be framed as an optimisation problem. Here Pinns can be used for solving the forward simulation and thus given as as well an approximated task for the same problem as in PINN_PDE, but framing it differently.

Furthermore, another field not shown here would be that PINNs can also be used for equation discovery, but not in a symbolic sense as done with Symbolic Regression and Model Discovery.
