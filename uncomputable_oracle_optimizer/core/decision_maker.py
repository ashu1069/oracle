import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt 
from .simulator import true_objective_function, collect_initial_data, halting_probability_function, uncomputable_oracle
from .brain import initialize_gp_regressor, initialize_gp_classifier, train_gp_regressor, train_gp_classifier, plot_models_and_data

# Implement Expected Improvement (EI)

def calculate_expected_improvement(x_candidates, gp_regressor, f_best):
    '''
    Calculates expected improvement from a set of candidate points

    Args:
        x_candidates (np.ndarray): array of candidate x values to evaluate
        gp_regressor (GaussianProcessRegressor): the trained GP model for f(x)
        f_best (float): the best observed objective so far (incumben solution)
    Returns:
        np.ndarray: Array of EI values for each candidate
    '''
    if gp_regressor is None or not hasattr(gp_regressor, 'X_train_') or len(gp_regressor.X_train_) == 0:
        # If no successful evaluations yet, return uniform high values for exploration
        return np.ones(len(x_candidates))

    # Predict mean and standard deviation for candidate points
    mu, sigma = gp_regressor.predict(x_candidates.reshape(-1,1), return_std=True)

    # Avoid division by zero  for sigma values that are very close to zero
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Z-score: how many standard deviations away from f_best
    Z = (mu - f_best) / sigma

    # Calculate EI
    ei_values = (mu-f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

    return np.maximum(0, ei_values)

# Implement Risk-Aware Acquisition Function (alpha(x))

def calculate_risk_aware_acquisition(x_candidates, gp_regressor, gp_classifier, f_best):
    '''
    Calculates the risk-aware acquisition function: P(halts|x) * ExpectedImprovement(x)
    Handles the case where the classifier is not fitted (e.g., only one class present) by using a default probability.
    '''
    # Calculate EI values using the regressor
    ei_values = calculate_expected_improvement(x_candidates, gp_regressor, f_best)

    # Predict halting probability using the classifier
    # If classifier isn't ready, assume 50% halt probability for exploration
    if (
        gp_classifier is None or
        not hasattr(gp_classifier, 'classes_') or
        len(getattr(gp_classifier, 'classes_', [])) < 2 or
        not hasattr(gp_classifier, 'scaler_') or
        not hasattr(gp_classifier, 'predict_proba')
    ):
        halt_probs = np.full(len(x_candidates), 0.5)
    else:
        try:
            x_candidates_scaled = gp_classifier.scaler_.transform(x_candidates.reshape(-1, 1))
            halt_probs = gp_classifier.predict_proba(x_candidates_scaled)[:, 1]
        except Exception as e:
            # If scaler or predict_proba fails, fallback to default
            halt_probs = np.full(len(x_candidates), 0.5)
    # Combine: P(halts|x) * EI(x)
    acquisition_values = halt_probs * ei_values
    return acquisition_values

# Example 
if __name__ == "__main__":
    print("--- Testing Phase 3 Components: Acquisition Function ---")

    # Re-use components from previous phases for demonstration
    x_domain_plot = np.linspace(0, 10, 200) # For plotting and candidate generation

    # 1. Collect some initial data
    print("\n--- Collecting Initial Data ---")
    initial_successful, initial_timed_out = collect_initial_data(
        num_samples=10, x_min=0.0, x_max=10.0, time_limit_per_eval=0.05, penalty_value=-1000.0
    )

    # 2. Train Surrogate Models
    print("\n--- Training Surrogate Models ---")
    gp_reg = initialize_gp_regressor()
    gp_reg = train_gp_regressor(gp_reg, initial_successful)
    
    gp_clf = initialize_gp_classifier()
    gp_clf = train_gp_classifier(gp_clf, initial_successful, initial_timed_out)

    # 3. Determine the best f_best found so far
    # If no successful evaluations, f_best should be a very low number
    f_best_so_far = max([s[1] for s in initial_successful]) if initial_successful else -np.inf
    print(f"Best f(x) found so far (f_best): {f_best_so_far:.2f}")

    # 4. Calculate the risk-aware acquisition function values for the entire domain
    x_candidates_eval = x_domain_plot # Use the plotting domain as our candidates
    acquisition_vals = calculate_risk_aware_acquisition(
        x_candidates_eval, gp_reg, gp_clf, f_best_so_far
    )

    print("\n--- Visualizing Acquisition Function ---")

    # Plotting the acquisition function
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_candidates_eval, acquisition_vals, color='purple', label='Risk-Aware Acquisition Function')
    ax.set_xlabel('x')
    ax.set_ylabel('Acquisition Value')
    ax.set_title('AI\'s Acquisition Function (Where to Probe Next)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the point with the maximum acquisition value
    best_x_to_probe = x_candidates_eval[np.argmax(acquisition_vals)]
    max_acq_val = np.max(acquisition_vals)
    ax.scatter(best_x_to_probe, max_acq_val, color='green', marker='*', s=200, 
               label=f'Next Probe (x={best_x_to_probe:.2f})', zorder=5)
    ax.annotate(f'Max Acq: {max_acq_val:.2f}\nx: {best_x_to_probe:.2f}',
                xy=(best_x_to_probe, max_acq_val), xytext=(best_x_to_probe + 0.5, max_acq_val * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05),
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="k", lw=1, alpha=0.8),
                ha='left')
    
    plt.tight_layout()
    plt.show()

    print("\nPhase 3 Complete. The AI can now decide where to probe next.")