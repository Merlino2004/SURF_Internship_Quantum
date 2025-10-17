import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skopt.space import Integer
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from sklearn.gaussian_process.kernels import Matern
import itertools 

from bayesian_optimization_quantum_AFM import QBO


class BayesianImageGenerator:
    def __init__(self, file_path, mode, model, n_start, n_end):
        """
        mode: 'Value' or 'Uncertainty'
        model: 'Quantum' or 'Classical'
        """
        self.file_path = file_path
        self.mode = mode
        self.model = model
        self.n_start = n_start
        self.n_end = n_end

        self.image = np.genfromtxt(file_path, delimiter=';', encoding='utf-8-sig')[1:251, 0:250]
        self.image = np.nan_to_num(self.image, nan=np.nanmean(self.image))
        self.max_voltage = np.max(self.image)

        self.dimensions = [
            Integer(0, self.image.shape[1] - 1, name='x_coordinate'),
            Integer(0, self.image.shape[0] - 1, name='y_coordinate')
        ]

        x_coords = np.arange(0, self.image.shape[1])
        y_coords = np.arange(0, self.image.shape[0])
        self.domain = np.array(list(itertools.product(x_coords, y_coords)))

    def objective_function(self, params):
        x, y = params
        x = int(np.clip(x, 0, self.image.shape[1] - 1))
        y = int(np.clip(y, 0, self.image.shape[0] - 1))
        return self.image[y, x]

    def synth_func(self, param, eps):
        device = AerSimulator()
        theta_image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image)) * np.pi

        if isinstance(param, (list, np.ndarray)) and len(param) == 2:
            j, i = int(param[0]), int(param[1])
            idx = i * 250 + j
        else:
            idx = int(param[0])
            i, j = divmod(idx, 250)


        theta = float(theta_image[i, j])

        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        tqc = transpile(qc, device)
        result = device.run(tqc, shots=1024).result()
        counts = result.get_counts()

        est = counts.get('1', 0) / 1024
        p = self.image[i, j]

        return est, p, 1024

    def prepare_grid(self):
        x_vals = np.linspace(0, self.image.shape[1] - 1, self.image.shape[1])
        y_vals = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        return X_grid, Y_grid, np.c_[X_grid.ravel(), Y_grid.ravel()]

    def run_optimizer(self):
        if self.model == "Quantum":
            random_features = {
                "s": np.random.randn(self.n_end, 2), 
                "b": np.random.uniform(0, 2 * np.pi, self.n_end),
                "obs_noise": 0.1,
                "v_kernel": 1.0
            }
            ts = np.arange(1, 1000)
            beta_t = 1 + np.sqrt(np.log(ts) ** 2)
            pbounds = {'x_coordinate': (0, self.image.shape[1] - 1), 'y_coordinate': (0, self.image.shape[0] - 1)}


            self.optimizer = QBO(
                f=self.synth_func,
                pbounds=pbounds,
                beta_t=beta_t,
                random_features=random_features,
                domain=self.domain,
                linear_bandit=False
            )

            self.x_list, self.y_list = self.optimizer.maximize(
                n_iter=500,
                init_points=self.n_end
            )

        elif self.model == "Classical":
            res = gp_minimize(
                func=self.objective_function,
                dimensions=self.dimensions,
                acq_func="EI",
                n_calls=self.n_end,
                n_random_starts=self.n_start,
                noise=0.0,
                random_state=1234
            )
            self.x_list = res.x_iters
            self.y_list = res.func_vals
            self.res = res
        else:
            raise ValueError(f"Chosen model ({self.model}) doesn't exist! Available models: Quantum & Classical")

    def plot_iteration(self, iteration):
        if not hasattr(self, "x_list"):
            raise RuntimeError("Optimizer not yet run. Call run_optimizer() first.")
        if iteration > len(self.x_list):
            raise ValueError(f"Iteration {iteration} exceeds available iterations ({len(self.x_list)}).")

        kernel = Matern(length_scale=10.0, nu=1.5)

        if self.model == "Quantum":
            X_train = np.array(self.x_list[:iteration]) 
            y_train = np.array(self.y_list[:iteration])

            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1 ** 2, n_restarts_optimizer=10) 
            gpr.fit(X_train, y_train)

            x_vals = np.linspace(0, self.image.shape[1] - 1, self.image.shape[1])
            y_vals = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
            X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
            X_predict = np.c_[X_grid.ravel(), Y_grid.ravel()]


            y_pred, sigma = gpr.predict(X_predict, return_std=True)

            predicted_image = y_pred.reshape(self.image.shape) * self.max_voltage
            sigma_image = sigma.reshape(self.image.shape)

        elif self.model == "Classical":
            X_train = np.array(self.x_list[:iteration])
            y_train = np.array(self.y_list[:iteration])
            X_grid, Y_grid, X_flat = self.prepare_grid()

            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=10)
            gpr.fit(X_train, y_train)

            y_pred, sigma = gpr.predict(X_flat, return_std=True)
            predicted_image = y_pred.reshape(X_grid.shape)
            sigma_image = sigma.reshape(X_grid.shape)
        else:
            raise ValueError(f"Chosen model ({self.model}) doesn't exist! Available models: Quantum & Classical")

        plt.figure(figsize=(8, 6))
        if self.mode == "Value":
            sns.heatmap(predicted_image, cmap='coolwarm', cbar_kws={'label': 'Predicted Voltage [V]'})
            plt.title(f'{self.model} — Predicted Surface (Iteration {iteration})')
        elif self.mode == "Uncertainty":
            sns.heatmap(sigma_image, cmap='viridis', cbar_kws={'label': 'Uncertainty (Std Dev)'})
            plt.title(f'{self.model} — Uncertainty (Iteration {iteration})')
        else:
            raise ValueError(f"Chosen mode ({self.mode}) doesn't exist! Available mode: Value & Uncertainty")

        plt.axis('off')

        best_idx = np.argmax(self.y_list[:iteration])
        if self.model == "Quantum":
            best_pixel_coords = self.x_list[best_idx]
            j_best, i_best = int(best_pixel_coords[0]), int(best_pixel_coords[1])
        else:
            j_best, i_best = self.x_list[best_idx]
        #plt.scatter(j_best, i_best, color='lime', marker='o', s=100, edgecolor='black', label='Best Pixel')
        plt.legend()
