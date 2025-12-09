import numpy as np
import dynesty
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import gamma
import itertools
import time

# Import from utils
from common.utils import safe_inverse, robust_sigmoid, dirichlet_transform, get_line_angle

class PreferenceSampler:
    """
    Bayesian Preference Learning Engine.
    Supports:
      - Algorithms: BAYES (MCMC), FTRL (MAP + Laplace)
      - Models: BT (Bradley-Terry), LIN (Linear)
      - Active: BALD, US
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
            Compute Laplace covariance (inverse negative Hessian).
            Uses direct inversion with jitter (approximation) as requested.
            """
            # --- Safety: Clip omega ---
            if alg == 'FTRL-LIN':
                omega_map += 1e-12
                omega_map /= np.sum(omega_map)
            
            n, d = self.X_diff.shape
            t = self.X_diff.dot(omega_map)
            
            # Determine Weights and Feature Matrix for Hessian
            if alg == 'FTRL-BT':
                p = expit(t)
                p = np.clip(p, 1e-12, 1-1e-12)
                W = p * (1.0 - p)
                prior_diag = (self.gamma_alpha_ftrl_bt - 1.0) / (omega_map ** 2)
                # BT uses raw difference vectors
                X_outer = self.X_diff 
                
            elif alg == 'FTRL-LIN':
                p = 0.5 * (1.0 + t)
                p = np.clip(p, 1e-12, 1-1e-12)
                W = 1 / (p**2)
                prior_diag = self.lambda_dirichlet_ftrl_lin / (omega_map ** 2)
                # LIN uses transformed vectors: 0.5 * (1 + diff)
                X_outer = 0.5 * (1.0 + self.X_diff)
                
            else: # Fallback
                print('Unknown algorithm for Laplace covariance.')

            # Accumulate Fisher Information Matrix F = X.T @ W @ X
            dF = np.einsum('i,ij,ik->jk', W, X_outer, X_outer)
            F = dF + np.diag(prior_diag)

            # Direct Inversion with Jitter
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
                    FTRL-LIN samples ARE clipped to Simplex.
        """
        rng = np.random.default_rng()
        d = len(omega_map)
        
        try:
            raw_samples = rng.multivariate_normal(mean=omega_map, cov=Sigma, size=n_samples)
        except np.linalg.LinAlgError:
            print("Covariance matrix not positive definite.")
            
        if alg == 'FTRL-BT':
            # bald_smps_mc for FTRL-BT returns samples directly 
            # without clipping or normalization.
            return raw_samples
        else:
            # FTRL-LIN: Constrained to Simplex
            samples = np.clip(raw_samples, 1e-9, 1.0)
            samples = samples / np.sum(samples, axis=1, keepdims=True)
            return samples
        
    def bald_mi_linear_appr(self, omega_map, Sigma, candidates, model_type):
        # Vectors: (N_cand, D)
        idx_a = [c[0] for c in candidates]
        idx_b = [c[1] for c in candidates]
        vec_diff = self.X[idx_a] - self.X[idx_b]

        # Utilities: (N_cand, N_samples)
        t = vec_diff @ omega_map
        
        # 1. Compute Var(t) = x.T @ Sigma @ x
        # This is the base variance of the latent score t
        var_t = np.einsum('ij,jk,ik->i', vec_diff, Sigma, vec_diff)

        if model_type == 'BT': 
            p = expit(t)
            p = np.clip(p, 1e-12, 1-1e-12)
            # Correct: MI approx 0.5 * p(1-p) * Var(t)
            mi = 0.5 * p * (1.0 - p) * var_t
            
        elif model_type == 'LIN':  
            p = 0.5 * (1.0 + t)
            p = np.clip(p, 1e-12, 1-1e-12)
            
            # 2. Convert Var(t) to Var(p)
            # Since p = 0.5(1+t), Var(p) = 0.5^2 * Var(t) = 0.25 * var_t
            var_p = 0.25 * var_t
            
            # Correct: MI approx 0.5 * Var(p) / (p(1-p))
            mi = 0.5 * var_p / (p * (1.0 - p))
            
        return mi

    # ------------------------------------------------------------------
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
            angle = get_line_angle(H_marginal, mi)
            if angle < 1:
                return H_marginal
            else:
                return mi


    def suggest_next_pair(self, all_indices, alg, active_method, current_state, n_samples_mc=200):
            """
            Determines the next best pair using the PROVIDED current_state.
            """
            if current_state is None:
                raise ValueError("suggest_next_pair requires 'current_state'.")

            algo_type, model_type = alg.split('-')
            full_alg_name = alg

            # 1. Generate Candidates
            possible_pairs = list(itertools.combinations(all_indices, 2))
            seen = set(tuple(x) for x in self.prefs)
            candidates = [p for p in possible_pairs if p not in seen and (p[1], p[0]) not in seen]
            
            if not candidates: return None
            
            # 2. Calculate Scores
            if algo_type == 'BAYES':
                samples = current_state
                scores = self._calculate_scores(candidates, samples, model_type, active_method)
            elif algo_type == 'FTRL':
                if active_method == 'US':
                    # US: Use MAP point only
                    samples = np.atleast_2d(current_state)
                    scores = self._calculate_scores(candidates, samples, model_type, active_method)
                elif active_method == 'BALD' or active_method == 'BALD+US': 
                    # BALD: Need Laplace Sampling
                    omega_map = current_state
                    Sigma = self.compute_laplace_covariance(omega_map, alg=full_alg_name)
                    #samples = self.sample_laplace(omega_map, Sigma, alg=full_alg_name, n_samples=n_samples_mc)
                    #Use Taylor approximation for scores
                    scores = self.bald_mi_linear_appr(omega_map, Sigma, candidates, model_type)
            else:
                raise ValueError(f"Unknown Algo: {algo_type}")
            
            return candidates[np.argmax(scores)]