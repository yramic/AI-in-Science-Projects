import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from equinox import nn

class FCN(eqx.Module):
    "Fully-connected neural network, with 1 hidden layer"
    
    input_layer: nn.Linear
    hidden_layer: nn.Linear
    output_layer: nn.Linear

    def __init__(self, in_size, hidden_size, out_size, key):
        "Initialise network parameters"
        
        key1, key2, key3 = jr.split(key, 3)
        self.input_layer = nn.Linear(in_size, hidden_size, key=key1)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, key=key2)
        self.output_layer = nn.Linear(hidden_size, out_size, key=key3)

    def __call__(self, x):
        "Defines forward model"
        x = jax.nn.relu(self.input_layer(x))
        x = jax.nn.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x