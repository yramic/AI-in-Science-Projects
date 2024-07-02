import os
from pde_find import Num_PDE
from utils import load_data, create_3d_plot, create_plot,  create_video

# Load Data from folder 'Data'
os.chdir('Data')
npz_files = [f for f in os.listdir() if f.endswith('.npz')]
npz_files = sorted(npz_files)

pdes = []
for file in npz_files:
    pdes.append(load_data(file))

pde_1 = pdes[0] # PDE Problem Nr 1
pde_2 = pdes[1] # PDE Problem Nr 2
pde_3 = pdes[2] # PDE Problem Nr 3

"""
To initialize the PDEs the following options are given
- order:        (int) defines the order of derivatives included in the search space
- combined:     (bool) defines if a combination of derivatives are allowed - u_xy*u_x
- mixed:        (bool) defines if mixed derivative terms are allowed - u_xy
- downsample:   (bool) defines if data should be downsampled in all dimensions
- alpha:        (float) defines the alpha value used for regularization in the linear
                regression. This is an important parameter since it defines the sparsity
                of our solution.
"""

# Initialization with best alpha values found through a search
pde_1 = Num_PDE(pde_1, order=3, alpha=1e-5)
pde_2 = Num_PDE(pde_2, order=3, alpha=1e-6)
pde_3 = Num_PDE(pde_3, order=2, mixed=False, downsample=True, alpha=0.0003)


"""
With create_plot there is also the option to plot any derivative
A feature for all plots is to set save=True, if the images want
to be stored
"""

name = 'pde1'
create_plot(pde_1.pde, pde_1.u_t, save=False, name=name)
name = 'pde1_3D'
create_3d_plot(pde_1.pde, save=False, name=name)

"""
Next, we want to build the library and present what assumptions were chosen for this 
first task:
"""

Theta, library = pde_1.build_library()
print(f'The dimensions of Theta under the assumptions chosen initially are: \n{Theta.shape}\n')
pde_1.print_search_space(library) # print search space for unidimensional PDE

xi, _ = pde_1.pde_search(Theta, features=True, find_alpha=True)

pred_u_t = Theta @ xi
true_u_t = pde_1.u_t.flatten()
residual = (true_u_t - pred_u_t).reshape(pde_1.u_t.shape[0], pde_1.u_t.shape[1])

import ipdb; ipdb.set_trace()