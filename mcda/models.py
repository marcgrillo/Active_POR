import numpy as np

class PiecewiseLinearTransformer:
    """
    Transforms raw decision matrix into utility feature vectors.
    """
    def __init__(self, ch_p):
        """
        Args:
            ch_p (list of arrays): Characteristic points (breakpoints) for each criterion.
        """
        self.ch_p = ch_p
        self.num_criteria = len(ch_p)
        self.n_w = [len(cp) - 1 for cp in ch_p]
        self.total_params = sum(self.n_w)

    @classmethod
    def from_equal_intervals(cls, data_matrix, num_intervals):
        """
        Alternative constructor that generates breakpoints automatically based on 
        equal intervals (Min to Max), matching your original `alternative_to_vector` logic.
        """
        mini = np.min(data_matrix, axis=0)
        maxi = np.max(data_matrix, axis=0)
        ch_p = []
        
        for j in range(data_matrix.shape[1]):
            # Create breakpoints: min, min+step, ..., max
            breakpoints = np.linspace(mini[j], maxi[j], num_intervals + 1)
            ch_p.append(breakpoints)
            
        return cls(ch_p)

    def transform(self, data_matrix):
        """
        Vectorized transformation.
        Returns: np.ndarray (N_alternatives, Total_Parameters)
        """
        N, M = data_matrix.shape
        features = np.zeros((N, self.total_params))
        current_col_idx = 0
        
        for j in range(M):
            col_data = data_matrix[:, j]
            breakpoints = self.ch_p[j]
            n_segments = len(breakpoints) - 1
            
            for k in range(n_segments):
                bp_low = breakpoints[k]
                bp_high = breakpoints[k+1]
                val = (col_data - bp_low) / (bp_high - bp_low)
                features[:, current_col_idx + k] = np.clip(val, 0, 1)
                
            current_col_idx += n_segments
            
        return features