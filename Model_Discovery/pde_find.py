import numpy as np
from collections import Counter
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from pde_library_utils import get_library_terms, initialize_derivatives

class Num_PDE:
    def __init__(self, pde, combined=False, order=3, mixed=True, downsample=False, alpha=1e-4):
        """
        Initialize the class with relevant derivatives given a pde passed as a 
        library.
        - pde:          (dict) with all measurements for the PDE problem
        - combined:     (bool) are combined derivative multiplications allowed 
        - order:        (int - 2 or 3) what should be the order of derivatives used
        - mixed:        (bool) include or neglect mixed derivative terms
        - downsample:   (bool) should the pde data be downsampled
        - alpha:        (float) hyperparameter for the ridge regression / lasso
        """
        if downsample:
            self.sampling_rate = 2
            pde = {key: self.downsample_dataset(value) for key, value in pde.items()}

        self.pde = pde
        self.u = pde['u']
        self.t = pde['t']
        self.x = pde['x']


        self.order = order # relevant for the library describes the order of derivatives involved
        self.mixed_derivatives = mixed # allow or neglect mixed derivatives in the library
        self.combined_derivatives = combined
        self.alpha = alpha

        if len(self.pde.keys()) == 5:
            # PDE Problem 3
            self.problem_dim = 2
            self.v = pde['v']
            self.y = pde['y']
            self.dx, self.dy, self.dt = self.find_di()
            self.di_args = (self.dx, self.dy, self.dt)

        elif len(self.pde.keys()) == 3:
            # PDE Problem 1 and 2
            self.problem_dim = 1
            self.dx, self.dt = self.find_di()
            self.di_args = (self.dx, self.dt)

        else:
            raise ValueError("The PDE Dictionary is not valid!")
        
        
        (self.u_t, self.u_x, self.u_xx, self.u_xxx, self.u_xxxx) = \
            initialize_derivatives((self.dx, self.dt), self.u, \
                                   problem_dim=1, deriv=True)

        if self.problem_dim == 1:
            (self.deriv_args, self.deriv_list) = \
                initialize_derivatives(self.di_args, self.u, v=None, \
                                       problem_dim=self.problem_dim, deriv=False)
            
        if self.problem_dim == 2:
            (self.deriv_args, self.deriv_list) = \
                initialize_derivatives(di_args=self.di_args, u=self.u, v=self.v, \
                                       problem_dim=self.problem_dim, \
                                       deriv=False)

        # derivative terms defined for a cominatorial setup and if a combination of derivatives is allowed
        self.derivatives = { self.deriv_list[i]: self.deriv_args[i] for i in range(len(self.deriv_args)) }
        # eliminate 'v_t' from derivatives:
        if 'v_t' in self.derivatives:
            del self.derivatives['v_t']

        # Define arguments to generate the pde library!
        if self.problem_dim == 1:
            if self.order == 2:
                args = (self.u, self.u_x, self.u_xx)
            elif self.order == 3:
                args = (self.u, self.u_x, self.u_xx, self.u_xxx)

        elif self.problem_dim == 2:
            # unpack relevant terms
            self.u_x, self.u_xx, self.u_xxx, self.v_x, self.v_xx, self.v_xxx, \
            self.u_y, self.u_yy, self.u_yyy, self.v_y, self.v_yy, self.v_yyy, \
            self.u_xy, self.u_xxy, self.u_xyy, self.v_xy, self.v_xxy, self.v_xyy, \
            self.v_t = self.deriv_args

            if self.order == 2:
                if self.mixed_derivatives:
                    args = (self.u, self.u_x, self.u_xx, \
                            self.v, self.v_x, self.v_xx, \
                            self.u_y, self.u_yy, self.v_y, self.v_yy, \
                            self.u_xy, self.v_xy)
                else:
                    args = (self.u, self.u_x, self.u_xx, \
                            self.v, self.v_x, self.v_xx, \
                            self.u_y, self.u_yy, self.v_y, self.v_yy)
            elif self.order == 3:
                if self.mixed_derivatives:
                    args = (self.u, self.u_x, self.u_xx, self.u_xxx, \
                            self.v, self.v_x, self.v_xx, self.v_xxx, \
                            self.u_y, self.u_yy, self.u_yyy, \
                            self.v_y, self.v_yy, self.v_yyy, \
                            self.u_xy, self.u_xxy, self.u_xyy, \
                            self.v_xy, self.v_xxy, self.v_xyy)
                else:
                    args = (self.u, self.u_x, self.u_xx, self.u_xxx, \
                            self.v, self.v_x, self.v_xx, self.v_xxx, \
                            self.u_y, self.u_yy, self.u_yyy, \
                            self.v_y, self.v_yy, self.v_yyy)
                
        self.args = args

        self.terms, self.library = \
            get_library_terms(order=self.order, arg_arr=self.args, \
                              problem_dim=self.problem_dim, mixed=self.mixed_derivatives)
        

    def find_di(self):
        """
        returns dx, dy and dt for an equidistant mesh
        PDE Problems 1, 2 and 3 are all equidistant
        """
        if len(self.pde.keys()) == 3:
            dt = self.t[0,1] - self.t[0,0]
            dx = self.x[1,0] - self.x[0,0]
            return dx, dt
        elif len(self.pde.keys()) == 5:
            dt = self.t[0,0,1] - self.t[0,0,0]
            dx = self.x[1,0,0] - self.x[0,0,0]
            dy = self.y[0,1,0] - self.y[0,0,0]
            return dx, dy, dt
            
    def downsample_dataset(self, data):
        """
        downsamples the data in every dimension for a coupled PDE

        - data:     3D tensor to be downsampled
        - return:   Downsampled 3D tensor given a sampling rate
        """
        assert len(data.shape) == 3, "Downsampling only valid for a threedimensional tensor"
        return data[::self.sampling_rate+1, ::self.sampling_rate+1, ::self.sampling_rate+1]
    
    
    def build_library(self):
        """
        Build the library matrix Theta(u) with all candidate terms.

        - combined_derivatives:     are combinations of derivatives are allowed (boolean)
        - return:                   The library matrix Theta(u)
        """

        # case 1: nonlinear combination of derivatives are allowed
        if self.combined_derivatives:
            # Start with non-linear and linear terms
            terms = self.terms
            library = self.library

            for name_1, deriv in self.derivatives.items():
                terms.append(self.u * deriv)
                library.append('u' + '*' + name_1)
                for name_2, deriv2 in self.derivatives.items():
                    terms.append(deriv * deriv2)
                    library.append(name_1 + '*' + name_2)

            # Delete redundancy:
            redundancy = [np.sum(terms[i]) for i in range(len(terms))]

            # now find duplicates:
            count_vals = Counter(redundancy)
            duplicates = [value for value, count in count_vals.items() if count > 1]

            duplicate_idx = {str(value): [] for value in duplicates}
            for index, value in enumerate(redundancy):
                if value in duplicates:
                    duplicate_idx[str(value)].append(index)

            duplicate_keys = list(duplicate_idx.keys())
            # delete always the second value in the list of duplicates!
            del_idx = [duplicate_idx[key][1] for key in duplicate_keys]
            # now delete terms in reversed order
            del_idx.sort(reverse=True)

            for index in del_idx:
                del terms[index]
                del library[index]

        # Case 2: nonlinear combination of derivatives are not allowed
        else:
            terms = self.terms
            library = self.library
        
        # Flatten each term and stack them as columns in the library matrix
        Theta = np.column_stack([term.flatten() for term in terms])
        return Theta, library
    

    def normalize(self, Theta):
        """
        scales input data by subtracting the mean and dividing through the variance
        - input: The relevant library Theta
        - return: scaled Theta and u_t as well as the mean and the variance to redo the scaling
        """
        scaler_Theta = StandardScaler()
        scaler_u_t = StandardScaler()

        Theta_scaled = scaler_Theta.fit_transform(Theta)
        u_t_scaled = scaler_u_t.fit_transform(self.u_t.flatten().reshape(-1, 1)).flatten()

        if len(self.pde.keys()) == 5:
            scaler_v_t = StandardScaler()
            v_t_scaled = scaler_v_t.fit_transform(self.v_t.flatten().reshape(-1,1)).flatten()
            return u_t_scaled, v_t_scaled, Theta_scaled, scaler_Theta
        else:
            return u_t_scaled, Theta_scaled, scaler_Theta


    def find_best_alpha(self, Theta):
        """
        The goal is to find the best lasso parameters in particular alpha
        - input: Feature matrix Theta
        - return: best alpha value
        """
        if self.problem_dim == 1:
            params = {'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

        if self.problem_dim == 2:
            # For PDE 3 there are higher alpha values required
            params = {'alpha': [1e-3, 1e-2, 1e-1]}

        lasso = Lasso(fit_intercept=False, max_iter=10000)
        model = GridSearchCV(lasso, params, cv=5)
        model.fit(Theta, self.u_t.flatten())

        return model.best_params_['alpha']


    def pde_search(self, Theta, features=False, normalize=False, find_alpha=False):
        """
        Solve for the sparse vector xi using ridge regression.
        - param Theta:  The library matrix Theta(u) on the righthandside
        - features:     (boolean) if activated a feature selection method identifies 
                        most significant terms
        - normalize:    (boolean) if activated a StandardScaler is used to scale 
                        u_t and Theta
        - find_alpha:   (boolean)  if activated a search to find the best penalization 
                        value for the regression is done. Note for some datasets there are 
                        convergence issues. Thus if you run into problems use the 
                        default fault and set an educated guess manually.
        - return:       The sparse vector xi and the reduced feature list if features 
                        is activated (True)
        """
        if normalize:
            if len(self.pde.keys()) == 5:
                u_t_scaled, v_t_scaled, Theta_scaled, scaler_Theta = self.normalize(Theta)
                LHS = np.concatenate([u_t_scaled.reshape(-1,1), 
                                      v_t_scaled.reshape(-1,1)], axis=1)
            else:
                u_t_scaled, Theta_scaled, scaler_Theta = self.normalize(Theta)
                LHS = u_t_scaled

        else:
            # non scaled version outperforms scaled version!
            if len(self.pde.keys()) == 5:
                LHS = np.concatenate([self.u_t.flatten().reshape(-1,1), 
                                      self.v_t.flatten().reshape(-1,1)], axis=1)
            else:
                LHS = self.u_t.flatten()

        if find_alpha:
            alpha = self.find_best_alpha(Theta)
        else:
            alpha = self.alpha

        if features:
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
            lasso.fit(Theta, LHS)
            model = SelectFromModel(lasso, prefit=True)
            features = model.get_support(indices=True)
        
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)

        if normalize:
            lasso.fit(Theta_scaled, LHS)
            xi = lasso.coef_
            xi = scaler_Theta.inverse_transform(xi.reshape(1, -1)).flatten()
        
        else:
            lasso.fit(Theta, LHS)
            xi = lasso.coef_

        # hard thresholding to enforce sparsity
        threshold = 1e-2
        if self.problem_dim == 2:
            for i in range(len(xi)):
                xi[i][np.abs(xi[i]) < threshold] = 0
        else:
            xi[np.abs(xi) < threshold] = 0

        return xi, features
    

    def model_comparison(self, prediction, features):
        """
        Eliminates predicted indices that are not likely to be true due to 
        pre estimation done. Used was a feature selection method to identify 
        the most significant terms represented as indices of a library in 'features'.

        - prediction:   results of the pde_find method
        - features:     results from features selection method 
        - return:       resulting indices of the library
        """

        non_zero_idx = np.nonzero(prediction)[0]
        del_idx = []
        
        for idx in non_zero_idx:
            if not (idx in features):
                del_idx.append([idx])

        # delete in reversed order!
        del_idx.sort(reverse=True)
        for idx in del_idx:
            del non_zero_idx[idx]
        
        return non_zero_idx
    
    
    def compute_residual(self, Theta, xi):
        """
        - Theta:    Feature matrix defined by the library
        - xi:       Predicted results for the PDE
        """
        pred_u_t = Theta @ xi
        true_u_t = self.u_t.flatten()
        original_shape = self.u_t.shape
        residual = (true_u_t - pred_u_t).reshape(original_shape[0], original_shape[1])
        return residual
    

    def print_search_space(self, lib):
        """
        prints the resulting pde library to gain further insights about the
        dimension of the search space
        """

        print('Defined search space under some assumption:')

        for i in range(len(lib)):
            if i+1 < 10:
                print(f'{i+1}:    {lib[i]}')
            elif i+1 >= 10 and i+1 < 100:
                print(f'{i+1}:   {lib[i]}')
            elif i+1 >= 100 and i+1 < 1000:
                print(f'{i+1}:  {lib[i]}')
            else:
                raise Exception("The defined search space is too big!")
            
        print(f'Elements in the search space: {i+1}')
    

    def pde_find_algorithm(self, features=False, normalize=False, find_alpha=False, show_lib=False):
        """
        Some necessary commands for an optimized version of the Lasso regression
        in the pde_search method

        - features:     (boolean) if activated a feature selection method identifies 
                        most significant terms
        - normalize:    (boolean) if activated a StandardScaler is used to scale 
                        u_t and Theta
        - find_alpha:   (boolean)  if activated a search to find the best penalization 
                        value for the regression is done. Note for some datasets there are 
                        convergence issues. Thus if you run into problems use the 
                        default fault and set an educated guess manually.
        - return:       (str) with resulting PDEs
        """

        Theta, library = self.build_library()

        if show_lib:
            self.print_search_space(library)

        pred, reduction = self.pde_search(
            Theta, 
            features=features, 
            normalize=normalize, 
            find_alpha=find_alpha
            )

        if features:
            assert len(self.pde.keys()) != 5, \
                "Feature pre estimation is only possible for a lower dimensional problem"
            
            res_idx = self.model_comparison(pred, reduction)

            library_np = np.array(library)

            print('u_t = ', end='')
            for i in range(len(res_idx)):
                if i+1 != len(res_idx):
                    print(f'{pred[res_idx[i]]} * {library_np[res_idx[i]]}', end=' + ')
                else:
                    print(f'{pred[res_idx[i]]} * {library_np[res_idx[i]]}')

        else:
            if self.problem_dim == 1:
                nonzero_idx = np.nonzero(pred)[0]

                library_np = np.array(library)

                print('u_t = ', end='')
                for i in range(len(nonzero_idx)):
                    if i+1 != len(nonzero_idx):
                        print(f'{pred[nonzero_idx[i]]} * {library_np[nonzero_idx[i]]}', end=' + ')
                    else:
                        print(f'{pred[nonzero_idx[i]]} * {library_np[nonzero_idx[i]]}')
            
            elif self.problem_dim == 2:
                rows, cols = np.nonzero(pred)

                library_np = np.array(library)

                if len(rows) != 0:

                    print('u_t = ', end='')
                    
                    idx = 0
                    while rows[idx] != 1:
                        if rows[idx+1] != 1:
                            print(f'{pred[rows[idx], cols[idx]]} * {library_np[cols[idx]]}', end=' + ')
                        else:
                            print(f'{pred[rows[idx], cols[idx]]} * {library_np[cols[idx]]}')
                        idx += 1
                    print('v_t = ', end='')
                    while idx < len(rows):
                        if idx+1 != len(rows):
                            print(f'{pred[rows[idx], cols[idx]]} * {library_np[cols[idx]]}', end=' + ')
                        else:
                            print(f'{pred[rows[idx], cols[idx]]} * {library_np[cols[idx]]}')
                        idx += 1

                else:
                    print('u_t = 0 \nv_t = 0\nReduce alpha or activate find_alpha for better results')

