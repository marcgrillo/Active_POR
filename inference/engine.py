import numpy as np
import dynesty
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import gamma, pearsonr
import itertools
import time

# Import from utils
from common.utils import safe_inverse, robust_sigmoid, dirichlet_transform, get_line_angle

class PreferenceSampler:
    """
    Bayesian Preference Learning Engine.
    
    This class handles:
    1.  **Bayesian Inference**: Estimating the user's utility function (omega) from pairwise preferences.
        -   **BAYES**: Full MCMC sampling using Nested Sampling (Dynesty).
        -   **FTRL**: Fast approximation using MAP (Maximum A Posteriori) estimation + Laplace Approximation (Hessian-based uncertainty).
    2.  **Active Learning**: Suggesting the next best pair to show the user to maximize information gain.
        -   **US (Uncertainty Sampling)**: Choose pairs where the model is most uncertain (Entropy).
        -   **BALD (Bayesian Active Learning by Disagreement)**: Choose pairs that maximize Mutual Information between the prediction and model parameters (Reduces epistemic uncertainty).
    3.  **Models**:
        -   **BT (Bradley-Terry)**: Sigmoid-based probabilistic model (Soft consistency).
        -   **LIN (Linear)**: Linear probability model (Harder consistency).
    """
    def __init__(self, feature_matrix, preferences, n_params):
        self.X = feature_matrix
        self.prefs = np.array(preferences, dtype=int) if len(preferences) > 0 else np.empty((0, 2), dtype=int)
        self.n_params = n_params
        
        # Dirichlet Prior (BAYES-LIN)
        self.alpha_dirichlet_bayes_lin = np.ones(self.n_params) * 0.5 
        # Dirichlet Prior (FTRL-LIN)
        self.lambda_dirichlet_ftrl_lin = 0.1
        # Gamma Prior (BAYES-BT)
        self.gamma_alpha_bayes_bt = 1.0
        self.gamma_beta_bayes_bt = 1.0
        # Gamma Prior (FTRL-BT)
        self.gamma_alpha_ftrl_bt = 2.0
        self.gamma_beta_ftrl_bt = 1.0
        
        self._update_diff_vectors()

    def add_preference(self, up_idx, down_idx):
        new_pref = np.array([[up_idx, down_idx]], dtype=int)
        self.prefs = np.vstack([self.prefs, new_pref]) if self.prefs.size > 0 else new_pref
        self._update_diff_vectors()

    def _update_diff_vectors(self):
        if self.prefs.size == 0:
            self.X_diff = np.empty((0, self.n_params))
        else:
            self.X_diff = self.X[self.prefs[:, 0]] - self.X[self.prefs[:, 1]]

    def _check_inverse(self, A, A_inv):
            """Verifies if inversion was successful."""
            result = A @ A_inv
            return np.allclose(result, np.eye(A.shape[0]), atol=1e-3)
            
    def _get_alr_jacobian(self, w):
        """
        Computes the Nx(N-1) Jacobian of the softmax transformation evaluator at w.
        Used for mapping between unconstrained space and the probability simplex.
        """
        N = len(w)
        J = np.zeros((N, N - 1))
        for i in range(N - 1):
            for j in range(N - 1):
                J[i, j] = w[i] * ((1.0 if i == j else 0.0) - w[j])
        for j in range(N - 1):
            J[N - 1, j] = -w[N - 1] * w[j]
        return J

    # ------------------------------------------------------------------
    # Likelihood Functions
    # ------------------------------------------------------------------

    def log_likelihood_lin(self, omega):
        """Linear Model: P(a>b) = 0.5 * (1 + (u_a - u_b))."""
        # omega = np.maximum(omega, 1e-10) # Safety handled by callers
        utility_diff = self.X_diff @ omega
        probs = 0.5 * (1.0 + utility_diff)
        if np.any(probs <= 0): return -np.inf
        return np.sum(np.log(probs))

    def log_likelihood_bt(self, omega):
        """Bradley-Terry: P(a>b) = sigmoid(u_a - u_b)."""
        utility_diff = self.X_diff @ omega
        # log(sigmoid(x)) = -logaddexp(0, -x)
        return np.sum(-np.logaddexp(0, -utility_diff))

    # ------------------------------------------------------------------
    # 1. BAYES: Nested Sampling
    # ------------------------------------------------------------------

    def ptform_diri(self, u):
        return dirichlet_transform(u, self.alpha_dirichlet_bayes_lin)
    
    def ptform_gamma(self, u):
        """Prior transform for Dynesty sampling."""
        return gamma.ppf(u, a=self.gamma_alpha_bayes_bt, scale=self.gamma_beta_bayes_bt)

    def run_nested(self, model='LIN', nlive=500, dlz = 0.5):
        loglike = self.log_likelihood_bt if model == 'BT' else self.log_likelihood_lin
        ptform = self.ptform_gamma if model == 'BT' else self.ptform_diri
        
        sampler = dynesty.NestedSampler(
            loglikelihood=loglike,
            prior_transform=ptform,
            ndim=self.n_params,
            bound='multi',
            nlive=max(self.n_params * 5, nlive)
        )
        sampler.run_nested(print_progress=False, dlogz = dlz)
        return sampler.results.samples_equal()

    def run_hmc_sampler(self, model='BT', n_samples=2000, n_leapfrog=10, step_size=0.05, tune_steps=500):
        """
        Hamiltonian Monte Carlo sampling for BT model in unconstrained log-space.
        """
        if model != 'BT':
            raise ValueError("HMC sampler currently only implemented for BT model.")
        
        # We sample theta = log(w) where w ~ Gamma(alpha, beta)
        alpha = self.gamma_alpha_bayes_bt
        beta = self.gamma_beta_bayes_bt
        
        def energy(theta):
            w = np.exp(theta)
            # Log likelihood
            u_diff = self.X_diff @ w
            ll = np.sum(-np.logaddexp(0, -u_diff))
            # Log prior in theta space: log P(w) + sum(theta_i)
            # log P(w) = sum((alpha - 1)*log(w) - w/beta) = sum((alpha - 1)*theta - w/beta)
            lprior_theta = np.sum((alpha - 1)*theta - w/beta + theta)
            return -(ll + lprior_theta)

        def grad_energy(theta):
            w = np.exp(theta)
            u_diff = self.X_diff @ w
            # grad LL w.r.t w: X_diff.T @ (1 - sigmoid(u_diff))
            # Since sigmoid(x) = expit(x), 1 - sigmoid(x) = expit(-x)
            grad_ll_w = self.X_diff.T @ expit(-u_diff)
            # grad of energy w.r.t theta_i = - [ grad_ll_w_i * w_i + alpha - w_i/beta ]
            return - (grad_ll_w * w + alpha - w / beta)

        rng = np.random.default_rng()
        # Initialize
        try:
            w_map = self.find_map(model=model)
            theta = np.log(np.clip(w_map, 1e-6, None))
        except Exception:
            theta = np.zeros(self.n_params)

        epsilon = step_size
        target_accept = 0.65
        
        # Tuning phase
        for t in range(tune_steps):
            p = rng.normal(0, 1, self.n_params)
            current_p = p.copy()
            current_theta = theta.copy()
            
            theta_prop = theta.copy()
            p_prop = p.copy()
            
            # Leapfrog
            grad_U = grad_energy(theta_prop)
            p_prop -= 0.5 * epsilon * grad_U
            for i in range(n_leapfrog):
                theta_prop += epsilon * p_prop
                grad_U = grad_energy(theta_prop)
                if i != n_leapfrog - 1:
                    p_prop -= epsilon * grad_U
            p_prop -= 0.5 * epsilon * grad_U
            
            current_U = energy(current_theta)
            prop_U = energy(theta_prop)
            
            current_K = 0.5 * np.sum(current_p**2)
            prop_K = 0.5 * np.sum(p_prop**2)
            
            # log acceptance probability
            log_accept = current_U - prop_U + current_K - prop_K
            accept_prob = np.exp(log_accept) if log_accept < 0 else 1.0
            if np.isnan(accept_prob):
                accept_prob = 0.0
                
            if rng.uniform() < accept_prob:
                theta = theta_prop
                
            # Adapt step size
            step = 1.0 / np.sqrt(t + 1)
            epsilon = epsilon * np.exp(step * (accept_prob - target_accept))
            epsilon = np.clip(epsilon, 1e-4, 0.5)

        # Sampling phase
        samples = []
        for _ in range(n_samples):
            p = rng.normal(0, 1, self.n_params)
            current_p = p.copy()
            current_theta = theta.copy()
            
            theta_prop = theta.copy()
            p_prop = p.copy()
            
            grad_U = grad_energy(theta_prop)
            p_prop -= 0.5 * epsilon * grad_U
            for i in range(n_leapfrog):
                theta_prop += epsilon * p_prop
                grad_U = grad_energy(theta_prop)
                if i != n_leapfrog - 1:
                    p_prop -= epsilon * grad_U
            p_prop -= 0.5 * epsilon * grad_U
            
            current_U = energy(current_theta)
            prop_U = energy(theta_prop)
            
            current_K = 0.5 * np.sum(current_p**2)
            prop_K = 0.5 * np.sum(p_prop**2)
            
            log_accept = current_U - prop_U + current_K - prop_K
            accept_prob = np.exp(log_accept) if log_accept < 0 else 1.0
            
            if not np.isnan(accept_prob) and rng.uniform() < accept_prob:
                theta = theta_prop
                
            samples.append(np.exp(theta.copy()))
            
        return np.array(samples)

    def run_mh_sampler(self, model='LIN', n_samples=2000, tune_steps=500, target_accept=0.3):
        """
        Metropolis-Hastings sampling with a log-normal proposal.
        """
        loglike = self.log_likelihood_bt if model == 'BT' else self.log_likelihood_lin
        
        def log_prior(w):
            if model == 'BT':
                # Gamma prior
                return np.sum(gamma.logpdf(w, a=self.gamma_alpha_bayes_bt, scale=self.gamma_beta_bayes_bt))
            else:
                # Dirichlet prior
                return np.sum((self.alpha_dirichlet_bayes_lin - 1) * np.log(w + 1e-12))
                
        def log_target(w):
            ll = loglike(w)
            if not np.isfinite(ll): return -np.inf
            lp = log_prior(w)
            if not np.isfinite(lp): return -np.inf
            return ll + lp

        rng = np.random.default_rng()
        # Initialize from MAP for FTRL model
        try:
            omega = self.find_map(model=model)
            omega = np.clip(omega, 1e-6, 1.0)
            if model == 'LIN':
                omega /= np.sum(omega)
        except Exception:
            omega = np.ones(self.n_params) / self.n_params
            
        current_target = log_target(omega)
        sigma = 0.1
        
        # 1. Tuning phase (Ghost MH)
        for t in range(tune_steps):
            # Lognormal proposal
            omega_prop = omega * np.exp(rng.normal(0, sigma, self.n_params))
            if model == 'LIN':
                omega_prop /= np.sum(omega_prop) # Project to simplex
            
            prop_target = log_target(omega_prop)
            
            if not np.isfinite(prop_target):
                log_accept_ratio = -np.inf
            else:
                correction = np.sum(np.log(omega_prop) - np.log(omega))
                log_accept_ratio = prop_target - current_target + correction
            
            accept_prob = min(1.0, np.exp(log_accept_ratio))
            
            if rng.uniform() < accept_prob:
                omega = omega_prop
                current_target = prop_target
                
            # Adapt sigma (Robbins-Monro stochastic approximation)
            step_size = 1.0 / np.sqrt(t + 1)
            sigma = sigma * np.exp(step_size * (accept_prob - target_accept))
            
        # 2. Sampling phase
        samples = []
        for _ in range(n_samples):
            omega_prop = omega * np.exp(rng.normal(0, sigma, self.n_params))
            if model == 'LIN':
                omega_prop /= np.sum(omega_prop)
                
            prop_target = log_target(omega_prop)
            
            if not np.isfinite(prop_target):
                log_accept_ratio = -np.inf
            else:
                correction = np.sum(np.log(omega_prop) - np.log(omega))
                log_accept_ratio = prop_target - current_target + correction
                
            if np.log(rng.uniform()) < log_accept_ratio:
                omega = omega_prop
                current_target = prop_target
                
            samples.append(omega.copy())
            
        return np.array(samples)

    # ------------------------------------------------------------------
    # 2. FTRL: Optimization (MAP)
    # ------------------------------------------------------------------

    def find_map(self, model='LIN'):
        if model == 'BT':
            return self._optimize_bt_cvxpy()
        else:
            return self._optimize_lin_scipy()

    def _optimize_bt_cvxpy(self):
        """
        Solves MAP for Bradley-Terry using Convex Optimization (CVXPY).
        Matches `frl_bt_omega_opt` from main_gpt.py.
        """
        omega = cp.Variable(self.n_params)
        
        # X_diff corresponds to (vec_data_prefs0 - vec_data_prefs1)
        # Utility diff = X_diff @ omega
        u_diff = self.X_diff @ omega
        
        # Log-Likelihood term: sum( log( sigmoid(u_diff) ) )
        # log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        # CVXPY has logistic(x) = log(1+exp(x)). 
        # So log(sigmoid(x)) = x - logistic(x)
        ll_term = cp.sum(u_diff - cp.logistic(u_diff))
        
        # Regularization (Gamma-like): alpha = 2, beta = 1 
        prior_term = cp.sum( (self.gamma_alpha_ftrl_bt - 1) * cp.log(omega) - self.gamma_beta_ftrl_bt * omega ) 

        # We want to MAXIMIZE LL + Reg, so MINIMIZE -(LL + Reg)
        objective = cp.Minimize( -ll_term - prior_term ) 
        
        constraints = [omega >= 1e-12]  # Non-negativity

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if omega.value is None: raise ValueError("Solver failed")
            return omega.value
        except:
            # Fallback
            return np.ones(self.n_params)/self.n_params

    def _optimize_lin_scipy(self):
        """Solves MAP for Linear Model using Scipy (Matches frl_lin_omega_opt)."""
        def neg_reg_ll(omega):
            ll = self.log_likelihood_lin(omega)
            # Reg: - lambda * sum(log(w))
            reg = self.lambda_dirichlet_ftrl_lin * np.sum(np.log(omega + 1e-12))
            return -(ll + reg) # Minimize negative

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(self.n_params)]
        x0 = np.ones(self.n_params) / self.n_params
        
        res = minimize(neg_reg_ll, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # ------------------------------------------------------------------
    # 3. FTRL: Laplace Approximation (Hessian)
    # ------------------------------------------------------------------

    def compute_laplace_covariance(self, omega_map, alg='FTRL-LIN', jitter=1e-8):
            """
            Computes the Laplace Approximation of the posterior covariance.
            
            The Laplace approximation approximates the posterior as a Gaussian centered at the MAP estimate.
            Covariance = Inverse(-Hessian of Log-Posterior)
            
            Mathematically:
            Sigma = (X^T @ W @ X + Prior_Precision)^-1
            
            Where:
            - W is the diagonal weight matrix derived from the second derivative of the link function (Sigmoid or Linear).
            - X is the differenced feature matrix.
            
            Args:
                omega_map (np.array): Maximum A Posteriori estimate of parameters.
                alg (str): 'FTRL-BT' or 'FTRL-LIN' to determine the link function.
                jitter (float): Small value added to diagonal for numerical stability during inversion.
            """
            # --- Safety: Clip omega (Linear model requires w > 0) ---
            if alg == 'FTRL-LIN':
                omega_map += 1e-12
                omega_map /= np.sum(omega_map)
            
            n, d = self.X_diff.shape
            t = self.X_diff.dot(omega_map) # Latent utility difference
            
            # Determine Weights (W) and Feature Matrix (X_outer) for Hessian
            if alg == 'FTRL-BT':
                p = expit(t)
                p = np.clip(p, 1e-12, 1-1e-12)
                # Second derivative of log-sigmoid is p(1-p)
                W = p * (1.0 - p)
                # Gamma prior diagonal approximation
                prior_diag = (self.gamma_alpha_ftrl_bt - 1.0) / (omega_map ** 2)
                # BT uses raw difference vectors
                X_outer = self.X_diff 
                
            elif alg == 'FTRL-LIN':
                p = 0.5 * (1.0 + t)
                p = np.clip(p, 1e-12, 1-1e-12)
                # Derivative of log(p) involves 1/p^2
                W = 1 / (p**2)
                # Dirichlet prior diagonal approximation
                prior_diag = self.lambda_dirichlet_ftrl_lin / (omega_map ** 2)
                # LIN uses transformed vectors: 0.5 * (1 + diff) ? No, actually derivative chain rule. 
                # This formulation follows the specific gradient derivation for the linear model.
                X_outer = 0.5 * (1.0 + self.X_diff)
                
            else: # Fallback
                print('Unknown algorithm for Laplace covariance.')

            # Accumulate Fisher Information Matrix F = X.T @ W @ X
            # einsum 'i,ij,ik->jk' performs the weighted outer product sum efficiently
            dF = np.einsum('i,ij,ik->jk', W, X_outer, X_outer)
            F = dF + np.diag(prior_diag) # Add prior precision (Regularization)

            # Direct Inversion with Jitter
            # We add jitter to ensure the matrix is Positive Definite before inversion
            if alg == 'FTRL-LIN':
                # Map to unconstrained space for simplex using ALR
                N = len(omega_map)
                J = self._get_alr_jacobian(omega_map)
                
                # F is the Hessian of the negative log posterior. So H_w = -F
                # Unconstrained Hessian H_y = - J.T @ H_w @ J = J.T @ F @ J
                H_y = J.T @ F @ J
                H_y += np.eye(N - 1) * jitter
                
                # Sigma_y = inv(H_y)
                try:
                    L = np.linalg.cholesky(H_y)
                    Sigma_y = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N - 1)))
                except np.linalg.LinAlgError:
                    Sigma_y = safe_inverse(H_y)
                
                if not self._check_inverse(H_y, Sigma_y):
                    print('Inversion of ALR Hessian was not successful. \n', H_y, Sigma_y)
                    time.sleep(5)
                return Sigma_y
            else:
                F += np.eye(d) * jitter
                Sigma = safe_inverse(F)
                
                if not self._check_inverse(F, Sigma):
                    print('Inversion was not successful. Printing F and Sigma: \n', F, Sigma)
                    time.sleep(5)
    
                return Sigma

    def sample_laplace(self, omega_map, Sigma, alg, n_samples=1000):
        """
        Generates samples from the Laplace approximation.
        
        Args:
            alg (str): 'FTRL-BT' or 'FTRL-LIN'.
                    FTRL-BT samples are NOT clipped/bounded.
                    FTRL-LIN samples use ALR unconstrained sampling and are mapped back to Simplex.
        """
        rng = np.random.default_rng()
        d = len(omega_map)
        
        if alg == 'FTRL-LIN':
            # FTRL-LIN: ALR unconstrained sampling
            y_map = np.log(omega_map[:-1]) - np.log(omega_map[-1])
            try:
                y_samples = rng.multivariate_normal(mean=y_map, cov=Sigma, size=n_samples)
            except np.linalg.LinAlgError:
                print("Covariance matrix not positive definite. Using purely MAP samples.")
                y_samples = np.tile(y_map, (n_samples, 1))
                
            # Map back to simplex: w = softmax( [y, 0] )
            y_samples_with_zero = np.hstack([y_samples, np.zeros((n_samples, 1))])
            max_y = np.max(y_samples_with_zero, axis=1, keepdims=True)
            exp_y = np.exp(y_samples_with_zero - max_y)
            samples = exp_y / np.sum(exp_y, axis=1, keepdims=True)
            
            # Ensure strict simplex boundaries numerically
            samples = np.clip(samples, 1e-9, 1.0)
            samples = samples / np.sum(samples, axis=1, keepdims=True)
            return samples
        else:
            # FTRL-BT: direct sampling without constraints
            try:
                raw_samples = rng.multivariate_normal(mean=omega_map, cov=Sigma, size=n_samples)
            except np.linalg.LinAlgError:
                print("Covariance matrix not positive definite. Using purely MAP samples.")
                raw_samples = np.tile(omega_map, (n_samples, 1))
            return raw_samples
        
    def bald_mi_linear_appr(self, omega_map, Sigma, candidates, model_type):
        """
        Analytic Linear Approximation of BALD Mutual Information.
        
        Instead of sampling parameters (MC), we approximate the variance of the utility difference 
        using the Laplace covariance matrix (Sigma) and propagate it to the MI score using a Taylor expansion.
        
        Faster than MC sampling but potentially less accurate for highly non-linear posterior/models.
        """
        # Vectors: (N_cand, D)
        idx_a = [c[0] for c in candidates]
        idx_b = [c[1] for c in candidates]
        vec_diff = self.X[idx_a] - self.X[idx_b]

        # Utilities: (N_cand,)
        t = vec_diff @ omega_map

        if model_type == 'BT': 
            p = expit(t)
            p = np.clip(p, 1e-12, 1-1e-12)
            # MacKay's Evidence Approximation / Probit Approximation for MI
            # MI approx 0.5 * p(1-p) * Var(t)
            # Intuition: Variance is highest when p is near 0.5, scaled by parameter uncertainty.
            # 1. Compute Var(t) = x.T @ Sigma @ x (Epistemic Uncertainty in latent space)
            # This measures how much the model is uncertain about the utility difference for this pair
            var_t = np.einsum('ij,jk,ik->i', vec_diff, Sigma, vec_diff)
            mi = 0.5 * p * (1.0 - p) * var_t
            
        elif model_type == 'LIN': 
            # The input Sigma is the covariance in the (N-1) unconstrained space
            p = 0.5 * (1.0 + t)
            p = np.clip(p, 1e-12, 1-1e-12)

            # Reconstruct Jacobian for ALR mapping
            J = self._get_alr_jacobian(omega_map)
            
            # Project N x N effective covariance: Sigma_eff = J @ Sigma_y @ J.T
            Sigma_eff = J @ Sigma @ J.T

            # Now calculate var_p using the effective covariance in w-space
            # Since p = 0.5 * (1 + delta_x @ omega), grad_w = 0.5 * vec_diff
            grad_w = 0.5 * vec_diff 
            var_p = np.einsum('ij,jk,ik->i', np.atleast_2d(grad_w), Sigma_eff, np.atleast_2d(grad_w))

            # Analytic MI for linear probability model
            # Note: We use the standard p(1-p) denominator here, though it is 
            # unstable at the boundaries (p=1) for linear models.
            mi = 0.5 * var_p / (p * (1.0 - p))
            
        return mi

    def _get_log_posterior_lin_vectorized(self, samples):
        """
        Calculates unnormalized log-posterior for a batch of samples (S, D).
        """
        # 1. Log-Likelihood
        # X_diff: (N_prefs, D), samples: (S, D)
        # utility_diff: (N_prefs, S)
        utility_diff = self.X_diff @ samples.T
        probs = 0.5 * (1.0 + utility_diff)
        
        # Handle boundaries
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        log_ll = np.sum(np.log(probs), axis=0) # (S,)
        
        # 2. Log-Prior (Dirichlet-like regularization used in FTRL-LIN)
        # reg = lambda * sum(log(w))
        log_prior = self.lambda_dirichlet_ftrl_lin * np.sum(np.log(samples + 1e-12), axis=1) # (S,)
        
        return log_ll + log_prior

    def _calculate_mi_weighted(self, candidates, samples, weights, model_type):
        """
        Calculates BALD MI scores using weighted samples (Importance Sampling).
        """
        # weights: (S,), normalized to sum to 1
        idx_a = [c[0] for c in candidates]
        idx_b = [c[1] for c in candidates]
        vec_diff = self.X[idx_a] - self.X[idx_b]
        
        # probs: (N_cand, S)
        if model_type == 'BT':
            u_diff = vec_diff @ samples.T
            probs = robust_sigmoid(u_diff)
        else:
            u_diff = vec_diff @ samples.T
            probs = 0.5 * (1 + u_diff)
            
        probs = np.clip(probs, 1e-9, 1-1e-9)
        
        # 1. H(Mean(P))
        # p_mean: (N_cand,)
        p_mean = np.sum(probs * weights, axis=1)
        H_marginal = - (p_mean * np.log(p_mean) + (1-p_mean) * np.log(1-p_mean))
        
        # 2. Mean(H(P))
        entropy_samples = - (probs * np.log(probs) + (1-probs) * np.log(1-probs))
        E_H_conditional = np.sum(entropy_samples * weights, axis=1)
        
        return H_marginal - E_H_conditional

    def bald_mi_importance_sampling(self, candidates, model_type, n_samples=5000, omega_map=None, Sigma=None):
        """
        Calculates BALD MI using Importance Sampling from a flat Dirichlet distribution.
        Falls back to Taylor approximation if ESS is too low (< 5%).
        """
        rng = np.random.default_rng()
        
        # 1. Draw samples from flat Dirichlet (alpha=1)
        samples = rng.dirichlet(np.ones(self.n_params), size=n_samples)
        
        # 2. Calculate unnormalized log-posterior and weights
        log_post = self._get_log_posterior_lin_vectorized(samples)
        
        # Normalize weights safely
        max_log = np.max(log_post)
        weights = np.exp(log_post - max_log)
        sum_w = np.sum(weights)
        
        if sum_w < 1e-20:
            print("Warning: All IS weights are zero. Falling back to Taylor.")
            return self.bald_mi_linear_appr(omega_map, Sigma, candidates, model_type)
            
        weights /= sum_w
        
        # 3. Calculate ESS
        ess = 1.0 / np.sum(weights**2)
        ess_percent = (ess / n_samples) * 100
        
        if ess_percent < 5.0:
            #print(f"ESS too low ({ess_percent:.1f}%), falling back to Taylor.")
            return self.bald_mi_linear_appr(omega_map, Sigma, candidates, model_type)
            
        # 4. Calculate MI
        return self._calculate_mi_weighted(candidates, samples, weights, model_type)
    # 4. Active Learning Logic (Unified)
    # ------------------------------------------------------------------

    def _calculate_scores(self, candidates, samples, model, method):
        """
        Computes acquisition scores (BALD or US).
        """
        # Vectors: (N_cand, D)
        idx_a = [c[0] for c in candidates]
        idx_b = [c[1] for c in candidates]
        vec_diff = self.X[idx_a] - self.X[idx_b]

        # Utilities: (N_cand, N_samples)
        u_diff = vec_diff @ samples.T
        
        # Probabilities
        if model == 'BT':
            probs = robust_sigmoid(u_diff)
        else:
            probs = 0.5 * (1 + u_diff)
        
        probs = np.clip(probs, 1e-9, 1-1e-9)
        
        # Entropy
        # H(p) = -p log p - (1-p) log (1-p)
        entropy_per_sample = - (probs * np.log(probs) + (1-probs) * np.log(1-probs))
        
        if method == 'US':
            # Uncertainty Sampling: Maximize Expected Entropy (Aleatoric + Epistemic)
            # Note: Sometimes US is defined as Entropy of Mean Prob. 
            # main_gpt.py `us_obj_func` -> `marg_entropy` which is Entropy(Mean(Prob))
            p_mean = np.mean(probs, axis=1)
            H_marginal = - (p_mean * np.log(p_mean) + (1-p_mean) * np.log(1-p_mean))
            return H_marginal
            
        elif method == 'BALD':
            # BALD: H(Mean(P)) - Mean(H(P))
            p_mean = np.mean(probs, axis=1)
            H_marginal = - (p_mean * np.log(p_mean) + (1-p_mean) * np.log(1-p_mean))
            E_H_conditional = np.mean(entropy_per_sample, axis=1)
            mi = H_marginal - E_H_conditional
            return mi
        
        elif method == 'BALD+US':
            # Calculate Conditional Entropy
            p_mean = np.mean(probs, axis=1)
            H_marginal = - (p_mean * np.log(p_mean) + (1-p_mean) * np.log(1-p_mean))
            E_H_conditional = np.mean(entropy_per_sample, axis=1)
            mi = H_marginal - E_H_conditional
            if len(H_marginal) < 2:
                return mi
                
            corr = np.corrcoef(H_marginal, mi)[0, 1]
            if np.isnan(corr): corr = 0.0
            
            if corr > 0:
                return mi
            else:
                return H_marginal


    def get_candidate_scores(self, all_indices, alg, active_method, current_state, n_samples_mc=2000, use_linear_approx=False, use_is_bald=False):
        """
        Determines the candidates and their scores using the PROVIDED current_state.
        """
        if current_state is None:
            raise ValueError("get_candidate_scores requires 'current_state'.")

        algo_type, model_type = alg.split('-')
        full_alg_name = alg

        # 1. Generate Candidates
        possible_pairs = list(itertools.combinations(all_indices, 2))
        seen = set(tuple(x) for x in self.prefs)
        candidates = [p for p in possible_pairs if p not in seen and (p[1], p[0]) not in seen]
        
        if not candidates: return [], []

        # 2a. Literature baselines (selection-only rules) are dispatched separately.
        from inference import baselines
        if active_method in baselines.BASELINE_METHODS:
            scores = baselines.score(self, candidates, active_method, full_alg_name,
                                     current_state, n_samples_mc)
            return candidates, scores

        # 2. Calculate Scores
        if algo_type == 'BAYES':
            samples = current_state
            scores = self._calculate_scores(candidates, samples, model_type, active_method)
        elif algo_type == 'FTRL':
            if active_method == 'US':
                # US: Use MAP point only
                samples = np.atleast_2d(current_state)
                scores = self._calculate_scores(candidates, samples, model_type, active_method)
            elif active_method in ['BALD', 'BALD+US']: 
                # BALD: Need Laplace Sampling or IS
                omega_map = current_state
                Sigma = self.compute_laplace_covariance(omega_map, alg=full_alg_name)
                
                if use_is_bald and model_type == 'LIN':
                    # Use Importance Sampling for FTRL-LIN
                    scores = self.bald_mi_importance_sampling(candidates, model_type, n_samples=5000, omega_map=omega_map, Sigma=Sigma)
                elif use_linear_approx:
                    scores = self.bald_mi_linear_appr(omega_map, Sigma, candidates, model_type)
                else:
                    samples = self.sample_laplace(omega_map, Sigma, alg=full_alg_name, n_samples=n_samples_mc)
                    scores = self._calculate_scores(candidates, samples, model_type, 'BALD')
        else:
            raise ValueError(f"Unknown Algo: {algo_type}")
        
        return candidates, scores


    # ------------------------------------------------------------------
    # 5. Tripartite Error Diagnostics
    # ------------------------------------------------------------------

    def _eval_neg_log_posterior(self, z, alg):
        """
        Evaluates the negative log-posterior F(z) = - (LL(z) + Prior(z)).
        For FTRL-LIN, z is the ALR-transformed unconstrained parameter (N-1).
        For FTRL-BT, z is the unconstrained parameter (N).
        """
        if alg == 'FTRL-LIN':
            # Map z (ALR) to omega (Simplex)
            z_with_zero = np.append(z, 0.0)
            max_z = np.max(z_with_zero)
            exp_z = np.exp(z_with_zero - max_z)
            omega = exp_z / np.sum(exp_z)
            omega = np.clip(omega, 1e-12, 1.0)
            omega /= np.sum(omega)
            
            # Neg LL
            u_diff = self.X_diff @ omega
            probs = 0.5 * (1.0 + u_diff)
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            ll = np.sum(np.log(probs))
            
            # Neg Prior
            prior = self.lambda_dirichlet_ftrl_lin * np.sum(np.log(omega))
            return -(ll + prior)
            
        elif alg == 'FTRL-BT':
            omega = np.clip(z, 1e-12, None)
            
            # Neg LL
            u_diff = self.X_diff @ omega
            ll = np.sum(-np.logaddexp(0, -u_diff))
            
            # Neg Prior (Gamma)
            prior = np.sum((self.gamma_alpha_ftrl_bt - 1) * np.log(omega) - self.gamma_beta_ftrl_bt * omega)
            return -(ll + prior)
            
        return 0.0

    def calculate_estimation_error_diagnostic(self, omega_map, Sigma, candidates, alg):
        """
        Calculates the FTRL estimation error diagnostic based on cubic distortion.
        Returns a dictionary with epsilon_t and diagnostics for each candidate.
        """
        try:
            L_bar = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # If not positive definite, no valid local geometry
            return {'epsilon_t': 1.0, 'cand_errors': {c: np.inf for c in candidates}}
            
        if alg == 'FTRL-LIN':
            z_hat = np.log(omega_map[:-1]) - np.log(omega_map[-1])
        else:
            z_hat = omega_map.copy()

        results = {'cand_errors': {}}
        
        if alg == 'FTRL-BT':
            # Compute analytic third directional derivative of the full objective
            dim = len(z_hat)
            D_j = np.zeros(dim)
            alpha = self.gamma_alpha_ftrl_bt
            probs = expit(self.X_diff @ omega_map)
            
            for j in range(dim):
                v_j = L_bar[:, j]
                
                if self.X_diff.size > 0:
                    x_v = self.X_diff @ v_j
                    term1 = np.sum(probs * (1.0 - probs) * (1.0 - 2.0 * probs) * (x_v ** 3))
                else:
                    term1 = 0.0
                    
                term2 = - 2.0 * (alpha - 1.0) * np.sum((v_j / omega_map) ** 3)
                D_j[j] = term1 + term2
                
            epsilon_t = (1.0 / 3.0) * np.max(np.abs(D_j))
            j_star = int(np.argmax(np.abs(D_j)))
            
            # Record extra outputs from todo.tex
            results['D_j'] = D_j
            results['j_star'] = j_star
            results['hessian_cond'] = np.linalg.cond(Sigma) # Cond of Sigma = Cond of H
            results['clipping_used'] = False
            results['bt_scaling'] = 'sum'
            
        else: # FTRL-LIN
            dim = len(z_hat)
            D_j = np.zeros(dim)
            h = 0.5
            
            for j in range(dim):
                v_j = L_bar[:, j]
                
                # Function along direction v_j
                def phi(s): return self._eval_neg_log_posterior(z_hat + s * v_j, alg)
                
                # Third derivative via finite differences
                T_vvv = (phi(2*h) - 2*phi(h) + 2*phi(-h) - phi(-2*h)) / (2 * h**3)
                D_j[j] = T_vvv
                
            epsilon_t = (1.0 / 3.0) * np.max(np.abs(D_j))
            j_star = int(np.argmax(np.abs(D_j)))
            
            # Record extra outputs from todo.tex
            results['D_j'] = D_j
            results['j_star'] = j_star
            results['hessian_cond'] = np.linalg.cond(Sigma)
            results['clipping_used'] = False
            results['lin_h'] = h

        results['epsilon_t'] = epsilon_t
        
        # Now compute the per-candidate diagnostics
        # We need the predictive variance of the response probability eta.
        # But we only need to provide the score components: Delta_geom and Delta_Tay.
        
        for cand in candidates:
            idx_a, idx_b = cand
            vec_diff = self.X[idx_a] - self.X[idx_b]
            
            if alg == 'FTRL-LIN':
                J = self._get_alr_jacobian(omega_map)
                grad_w = 0.5 * vec_diff
                grad_z = J.T @ grad_w
                var_eta = np.einsum('i,ij,j->', grad_z, Sigma, grad_z)
                t = vec_diff @ omega_map
                eta = 0.5 * (1.0 + t)
            else:
                t = vec_diff @ omega_map
                eta = expit(t)
                grad_w = vec_diff * eta * (1 - eta)
                var_eta = np.einsum('i,ij,j->', grad_w, Sigma, grad_w)
                
            eta = np.clip(eta, 1e-9, 1-1e-9)
            
            # Base analytic score
            B_LT = 0.5 * var_eta / (eta * (1 - eta)) if alg == 'FTRL-LIN' else 0.5 * var_eta / (eta * (1 - eta))
            
            if epsilon_t >= 1.0:
                results['cand_errors'][cand] = np.inf
            else:
                Delta_geom = B_LT * (epsilon_t / (1 - epsilon_t))
                var_eff = var_eta / (1 - epsilon_t)
                Tay_coef = (1 - 3*eta + 3*eta**2) / (4 * eta**3 * (1 - eta)**3)
                Delta_Tay = Tay_coef * (var_eff**2)
                results['cand_errors'][cand] = abs(Delta_geom) + abs(Delta_Tay)
                
        return results



    def calculate_bayes_estimation_error(self, candidates, samples, model_type, active_method, alpha=0.05):
        """
        Calculates the near-optimal-set diagnostic for BAYES models.
        """
        S = len(samples)
        # Number of batch-means batches for the MCSE of the acquisition gap.
        # Target ~20 batches, but keep a floor of 10 so the batch standard deviation
        # (ddof=1) is itself stable, and never allow fewer than 2 samples per batch.
        B = min(20, max(10, S // 100))
        B = min(B, S // 2)
        Q_t_size = len(candidates)

        # BALD+US is not a single fixed acquisition function: _calculate_scores picks
        # BALD or US per call from the sign of corr(H_marginal, MI), so across batches it
        # can flip. The resulting gap MCSE - and hence the saved estimation-error quantities
        # (|C_t|, rho_t, Delta_MC=E_size, mcse=E_sep) - are not meaningful for BALD+US.
        if active_method == 'BALD+US' and not getattr(PreferenceSampler, '_bald_us_esterr_warned', False):
            import warnings
            warnings.warn(
                "E_est (BAYES) with active_method='BALD+US': the acquisition type can flip "
                "between batches, so the saved columns E_size (Delta_MC) and E_sep (mcse), "
                "and the derived |C_t|/rho_t, are NOT reliable for BALD+US runs.",
                RuntimeWarning, stacklevel=2,
            )
            PreferenceSampler._bald_us_esterr_warned = True

        if B < 2 or Q_t_size <= 1:
            return {
                '|C_t|': Q_t_size, 'rho_t': 1.0,
                'Delta_MC': 0.0, 'c_t': 0.0, 'B': B, 'S': S, 'rho_t_1': True,
                'mcse_median': 0.0, 'mcse_max': 0.0
            }
            
        batch_size = S // B
        
        # Overall scores
        A_hat = self._calculate_scores(candidates, samples, model_type, active_method)
        q_star_idx = int(np.argmax(A_hat))
        
        G_hat = A_hat[q_star_idx] - A_hat
        
        batch_gaps = np.zeros((B, Q_t_size))
        
        for b in range(B):
            batch_samples = samples[b*batch_size : (b+1)*batch_size]
            b_scores = self._calculate_scores(candidates, batch_samples, model_type, active_method)
            b_score_q_star = b_scores[q_star_idx]
            batch_gaps[b, :] = b_score_q_star - b_scores
            
        mcse_G = np.std(batch_gaps, axis=0, ddof=1) / np.sqrt(B)
        
        # Fixed ~2-sigma confidence multiplier (decision: constant c_t = 2, not the
        # per-step multiple-comparison z_{1-alpha/(M_t-1)} nor a time-uniform schedule).
        c_t = 2.0
        
        delta_MC = c_t * mcse_G
        
        C_t_mask = G_hat <= delta_MC
        
        C_t_size = np.sum(C_t_mask)
        rho_t = C_t_size / Q_t_size
        
        if C_t_size > 0:
            Delta_MC = np.max(G_hat[C_t_mask])
        else:
            Delta_MC = 0.0
            
        return {
            '|C_t|': int(C_t_size),
            'rho_t': float(rho_t),
            'Delta_MC': float(Delta_MC),
            'c_t': float(c_t),
            'B': B,
            'S': S,
            'rho_t_1': bool(rho_t == 1.0),
            'mcse_median': float(np.median(mcse_G)),
            'mcse_max': float(np.max(mcse_G))
        }


    def suggest_next_pair(self, all_indices, alg, active_method, current_state, n_samples_mc=2000, use_linear_approx=False, use_is_bald=False):
        """
        Determines the next best pair using the PROVIDED current_state.
        """
        candidates, scores = self.get_candidate_scores(all_indices, alg, active_method, current_state, n_samples_mc, use_linear_approx, use_is_bald)
        if not candidates: return None
        return candidates[np.argmax(scores)]

