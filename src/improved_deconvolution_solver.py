import numpy as np
import cvxpy as cp

class ImprovedDeconvolutionSolver:
    """
    Improved deconvolution solver implementing branch-specific edge handling
    with proper mirroring and padding techniques.
    """

    def __init__(self, config, g, H, gamma, N=None, f_replication=None,
                 b=None, obj_error_mode='additive', kappa=5e-3):
        """
        Initialize the deconvolution solver.
        
        Args:
            config: Configuration object with branch/phase information
            g: Observed signal vector
            H: Convolution matrix
            gamma: Regularization parameter for wavelet smoothing
            N: Normalization matrix (default: identity)
            f_replication: Replication factors for each position (default: ones)
            b: Scaling factor (default: 1)
            obj_error_mode: 'additive' or 'multiplicative' error model
            kappa: Regularization parameter for CG1-DG1 similarity
        """
        n, m = H.shape

        if N is None:
            N = np.eye(n)

        if f_replication is None:
            f_replication = np.ones(m)

        if b is None:
            b = 1

        self.obj_error_mode = obj_error_mode
        self.config = config
        self.g = g
        self.H = H
        self.N = N
        self.f_replication = f_replication
        self.b = b
        self.kappa = kappa
        self.gamma = gamma
        self.verbose = False
        
    def deconvolve(self):
        """
        Perform deconvolution using branch-specific edge handling.
        """
        # Get positions for all branches and phases
        f_i = self.config.get_Hpositions_for_branch('i')
        f_t = self.config.get_Hpositions_for_branch('t')
        f_b = self.config.get_Hpositions_for_branch('b')
        f_cg1 = self.config.get_Hpositions_for_phase('CG1')
        f_rg1 = self.config.get_Hpositions_for_phase('RG1')
        f_dg1 = self.config.get_Hpositions_for_phase('DG1')
        f_postg1 = self.config.get_Hpositions_for_phase('postG1')
        
        # Matrix dimensions
        n, m = self.H.shape
        
        # Prepare final result vector
        f_final = np.zeros(m)
        
        # ============= (LEFT MIRRORING) =============
        
        # Create branch-specific edge handling
        # Recovery branch: left mirroring (for right edge)
        f_i_left_mirror = np.concatenate([np.flip(f_i), f_i])
        
        # Top branch: duplication for cyclicity
        f_t_duplication = np.concatenate([f_t, f_t])
        
        # Bottom branch: duplication for cyclicity
        f_b_duplication = np.concatenate([f_b, f_b])
        
        # Get wavelet kernels for each branch
        W_i = self._get_wavelet_kernel(len(f_i))
        W_t = self._get_wavelet_kernel(len(f_t))
        W_b = self._get_wavelet_kernel(len(f_b))
        
        # Create block matrix structures for wavelets
        W_i_padded = self._create_block_wavelet(W_i)
        W_t_padded = self._create_block_wavelet(W_t, mirror=False)  # No mirroring for duplication
        W_b_padded = self._create_block_wavelet(W_b, mirror=False)  # No mirroring for duplication
        
        # Run second optimization (top and bottom branches remain the same)
        f_left_mirror = self._run_optimization(
            f_i_left_mirror, f_t_duplication, f_b_duplication,
            W_i_padded, W_t_padded, W_b_padded,
            f_cg1, f_dg1
        )
        
        f_final = f_left_mirror
        
        # Store results
        self.f = f_final
        
        # Calculate norms
        self.rn = self._calculate_residual_norm(f_final)
        self.sn = self._calculate_smoothness_norm(f_final, f_i, f_t, f_b)
        
        if self.verbose:
            print("Fit norm:", self.rn)
            print("Smoothing norm:", self.sn)
            print("Recovery branch smoothing:", self._calculate_branch_smoothness(f_final, f_i))
            print("Top branch smoothing:", self._calculate_branch_smoothness(f_final, f_t))
            print("Bottom branch smoothing:", self._calculate_branch_smoothness(f_final, f_b))
            
        return f_final
    
    def _get_wavelet_kernel(self, length):
        """
        Get wavelet kernel of specified length.
        This is a placeholder for your actual wavelet kernel implementation.
        
        Args:
            length: Length of the wavelet kernel
            
        Returns:
            Wavelet kernel matrix
        """
        from src.helpers import get_wavelet_kernel
        return get_wavelet_kernel(length, par=5)
    
    def _create_block_wavelet(self, W, mirror=True):
        """
        Create block wavelet matrix with zero padding.
        
        Args:
            W: Original wavelet matrix
            mirror: Whether to mirror the bottom-right block (True) or
                   duplicate it (False)
            
        Returns:
            Block matrix with zero padding
        """
        n = W.shape[0]
        W_pad = np.zeros((n, n))
        
        # Create block matrix
        if mirror:
            # Use mirroring (flipped kernel)
            W_block = np.block([
                [W, W_pad],
                [W_pad, np.fliplr(W)]
            ])
        else:
            # Use duplication (same kernel)
            W_block = np.block([
                [W, W_pad],
                [W_pad, W]
            ])
        
        return W_block
    
    def _run_optimization(self, f_i_indices, f_t_indices, f_b_indices, 
                          W_i, W_t, W_b, f_cg1, f_dg1):
        """
        Run a single optimization pass with the given edge handling configuration.
        
        Args:
            f_i_indices: Indices for recovery branch (with mirroring)
            f_t_indices: Indices for top branch (with duplication)
            f_b_indices: Indices for bottom branch (with duplication)
            W_i: Wavelet kernel for recovery branch
            W_t: Wavelet kernel for top branch
            W_b: Wavelet kernel for bottom branch
            f_cg1: Indices for CG1 phase
            f_dg1: Indices for DG1 phase
            
        Returns:
            Optimized f vector
        """
        f_i = self.config.get_Hpositions_for_branch('i')
        f_t = self.config.get_Hpositions_for_branch('t')
        f_b = self.config.get_Hpositions_for_branch('b')
        f_cg1 = self.config.get_Hpositions_for_phase('CG1')
        f_rg1 = self.config.get_Hpositions_for_phase('RG1')
        f_dg1 = self.config.get_Hpositions_for_phase('DG1')
        f_postg1 = self.config.get_Hpositions_for_phase('postG1')

        # Matrix dimensions
        n, m = self.H.shape
        
        # Variables for optimization
        f = cp.Variable(m)
        f_baseline = cp.Variable(1)
        
        # Model a baseline value, so smoothing constraints are applied to variations on the baseline
        f_non_replicative = (f + f_baseline)
        f_combined = cp.multiply(f_non_replicative, self.f_replication)
        
        # Error term calculation
        eps = 1e-5
        if self.obj_error_mode == 'multiplicative':
            elementwise_result = (self.N @ self.H @ f_combined * self.b) / (self.g + eps) - 1
        else:  # 'additive'
            elementwise_result = (self.N @ self.H @ f_combined * self.b + eps) - (self.g + eps)
        
        # Fit error term
        fit_norm_result = cp.square(cp.norm(elementwise_result, 2))
        
        # Smoothing terms for each branch
        smooth_f_i_result = W_i @ (f[f_i_indices])
        smooth_f_t_result = W_t @ (f[f_t_indices]) 
        smooth_f_b_result = W_b @ (f[f_b_indices])
        
        # Scale factors based on branch proportions (recovery is shortest, bottom is longest)
        # You may need to adjust these weights
        i_weight = 1  # Recovery branch
        t_weight = 1  # Top branch
        b_weight = 1  # Bottom branch (longest)
        
        # Combined smoothing result with weights
        smooth_result = (i_weight * cp.sum(cp.abs(smooth_f_i_result)) + 
                         t_weight * cp.sum(cp.abs(smooth_f_t_result)) + 
                         b_weight * cp.sum(cp.abs(smooth_f_b_result)))
        
        # CG1-DG1 regularization 
        tb_regularization_result = f[f_dg1] - f[f_cg1]
        cg1_dg1_regularization_result = cp.sum(cp.abs(tb_regularization_result))
        
        # Objective function
        objective = cp.Minimize(
            fit_norm_result + 
            self.gamma * smooth_result# +
            #self.kappa * cg1_dg1_regularization_result
        )
        
        # Constraints
        constraints = [
            f >= 0,  # Non-negativity
            f_baseline >= 0,
            f[f_i[0]] == f[f_t[-1]+1]  # Halted cells constraint if needed
        ]
        
        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)  # Adjust solver as needed
        
        return f.value + f_baseline.value
    
    def _calculate_residual_norm(self, f):
        """
        Calculate the residual norm (fit error).
        
        Args:
            f: Deconvolved signal
        
        Returns:
            Residual norm
        """
        f_combined = f * self.f_replication
        pred_g = self.H @ f_combined * self.b
        
        eps = 1e-5
        if self.obj_error_mode == 'multiplicative':
            elementwise_result = pred_g / (self.g + eps) - 1
        else:  # 'additive'
            elementwise_result = pred_g - self.g
            
        return np.sum(elementwise_result**2)
    
    def _calculate_branch_smoothness(self, f, branch_indices):
        """
        Calculate smoothness for a specific branch.
        
        Args:
            f: Deconvolved signal
            branch_indices: Indices for the branch
            
        Returns:
            Branch smoothness norm
        """
        W = self._get_wavelet_kernel(len(branch_indices))
        smooth_branch = W @ f[branch_indices]
        return np.sum(np.abs(smooth_branch))
    
    def _calculate_smoothness_norm(self, f, f_i, f_t, f_b):
        """
        Calculate the overall smoothness norm.
        
        Args:
            f: Deconvolved signal
            f_i: Recovery branch indices
            f_t: Top branch indices
            f_b: Bottom branch indices
            
        Returns:
            Overall smoothness norm
        """
        i_smoothness = self._calculate_branch_smoothness(f, f_i)
        t_smoothness = self._calculate_branch_smoothness(f, f_t)
        b_smoothness = self._calculate_branch_smoothness(f, f_b)
        
        # Weight factors matching those in optimization
        i_weight = 1
        t_weight = 1
        b_weight = 1
        
        return i_weight * i_smoothness + t_weight * t_smoothness + b_weight * b_smoothness
    
    def plot_fit(self, plot_timepoints=False):
        """
        Plot the fit results.
        
        Args:
            plot_timepoints: Whether to plot by timepoints or indices
        """
        from matplotlib import pyplot as plt

        config = self.config
        i_indices = config.get_Hpositions_for_branch('i')
        t_indices = config.get_Hpositions_for_branch('t')
        b_indices = config.get_Hpositions_for_branch('b')

        i_tps = config.get_timepoints_for_branch('i')
        t_tps = config.get_timepoints_for_branch('t')
        b_tps = config.get_timepoints_for_branch('b')

        if not plot_timepoints:
            # Plot by indices
            i_tps = np.arange(len(i_tps))
            t_tps = np.arange(len(t_tps))
            b_tps = np.arange(len(b_tps))

        num_cols = 4
        fig, axs = plt.subplots(1, num_cols, figsize=(16, 3))

        g = self.g
        f = self.f

        max_value = np.concatenate([g, f]).max()
        ylims = -((max_value*0.05)), (max_value*1.05)

        gamma_predicted_g = self.H @ (f * self.f_replication) * self.b

        ax_row = axs

        ax = ax_row[0]
        ax.plot(g[:], c='black', lw=3, label="Raw data")
        ax.plot(gamma_predicted_g, c='red',
                lw=3, label="Optimal $\\gamma$ solution")
        ax.set_title("Data vs Fit")
        ax.legend()
        ax.set_ylim(*ylims)

        ax = ax_row[1]
        ax.plot(i_tps, f[i_indices], c='red', lw=3)
        ax.set_title("Recovery branch")
        ax.set_ylim(*ylims)

        ax = ax_row[2]
        ax.plot(b_tps, f[b_indices], c='blue', lw=3, alpha=0.25)
        ax.plot(t_tps, f[t_indices], c='red', lw=3)
        ax.set_ylim(*ylims)
        ax.set_title("Top branch")

        ax = ax_row[3]
        ax.plot(t_tps, f[t_indices], c='red', lw=3, alpha=0.25)
        ax.plot(b_tps, f[b_indices], c='blue', lw=3)
        ax.set_ylim(*ylims)
        ax.set_title("Bottom branch")
        
        plt.tight_layout()
        return fig, axs
        
    def plot_edge_handling_comparison(self):
        """
        Plot comparison of left and right mirroring solutions for each branch.
        Shows how the final solution combines the best parts of each.
        """
        from matplotlib import pyplot as plt

        config = self.config
        i_indices = config.get_Hpositions_for_branch('i')
        t_indices = config.get_Hpositions_for_branch('t')
        b_indices = config.get_Hpositions_for_branch('b')

        fig, axs = plt.subplots(3, 1, figsize=(12, 9))
        
        # Recovery branch
        ax = axs[0]
        ax.plot(self.f[i_indices], 'k--', linewidth=2, label='Combined solution')
        ax.set_title("Recovery Branch Edge Handling")
        ax.legend()
        ax.set_xlabel("Position")
        ax.set_ylabel("Expression")
        ax.set_ylim(0, self.f.max()*1.2)

        # Top branch
        ax = axs[1]
        ax.plot(self.f[t_indices], 'k--', linewidth=2, label='Combined solution')
        ax.set_title("Top Branch Edge Handling")
        ax.legend()
        ax.set_xlabel("Position")
        ax.set_ylabel("Expression")
        ax.set_ylim(0, self.f.max()*1.2)

        # Bottom branch
        ax = axs[2]
        ax.plot(self.f[b_indices], 'k--', linewidth=2, label='Combined solution')
        ax.set_title("Bottom Branch Edge Handling")
        ax.legend()
        ax.set_xlabel("Position")
        ax.set_ylabel("Expression")
        ax.set_ylim(0, self.f.max()*1.2)
        
        plt.tight_layout()
        return fig, axs
