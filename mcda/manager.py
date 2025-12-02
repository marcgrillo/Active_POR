import numpy as np
import pandas as pd
from mcdm_models import PiecewiseLinearTransformer
from inference import PreferenceSampler

class RankingSystem:
    def __init__(self, df_alternatives, criteria_breakpoints):
        self.df = df_alternatives.copy()
        self.names = self.df.iloc[:, 0].values
        self.raw_data = self.df.iloc[:, 1:].values.astype(float)
        
        self.transformer = PiecewiseLinearTransformer(criteria_breakpoints)
        self.feature_matrix = self.transformer.transform(self.raw_data)
        
        self.sampler = PreferenceSampler(
            self.feature_matrix, 
            preferences=[], 
            n_params=self.transformer.total_params
        )

    def add_preference(self, preferred_name, other_name):
        idx_up = np.where(self.names == preferred_name)[0][0]
        idx_down = np.where(self.names == other_name)[0][0]
        self.sampler.add_preference(idx_up, idx_down)

    def run_inference(self):
        self.omega_samples = self.sampler.run_nested()

    def get_ranking_scores(self):
        if not hasattr(self, 'omega_samples'):
            self.run_inference()
        utilities = self.feature_matrix @ self.omega_samples.T
        mean_u = np.mean(utilities, axis=1)
        return pd.DataFrame({'Alternative': self.names, 'Mean Utility': mean_u}).sort_values('Mean Utility', ascending=False)