import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from .simulator import uncomputable_oracle, true_objective_function, halting_probability_function, collect_initial_data

# Initialize and Train Gaussian Process Regressor (MF)
def initialize_gp_regressor():
    '''Initialize a Gaussian Process Regressor for the objective function
    '''
    # Kernel: captures the 'smoothness' and overall scale of the function
    # ConstantKernel: scales the output of the Matern kernel
    # Matern: a common kernel for functions with varying smoothness
    # WhiteKernel: adds noise to the observations (essential for GP regression)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale = 1.0, length_scale_bounds = (1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5,1e0))

    gp_regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10)

    return gp_regressor

def train_gp_regressor(gp_regressor, successful_evals):
    '''
    Trains the GP regressor using only the data from successful (halted) evaluations
    '''
    if not successful_evals:
        print("Warning: No successful evaluations yet to train GP regressor")
        return gp_regressor

    X_train = np.array([s[0] for s in successful_evals]).reshape(-1,1)
    y_train = np.array([s[1] for s in successful_evals])

    gp_regressor.fit(X_train, y_train)
    return gp_regressor

# Initialize and Train Gaussian Process Classifier (MH)

def initialize_gp_classifier():
    '''
    Initializes a Gaussian Process Classifier for the halting probability. GPC models the probability of a binary outcome (halted vs. timed out)
    '''
    # Kernel for GPC: can be simpler than for regression, often just RBF
    # Use ConstantKernel with RBF
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) + Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)

    gp_classifier = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10)

    gp_classifier.scaler_ = StandardScaler()
    return gp_classifier

def train_gp_classifier(gp_classifier, successful_evals, timed_out_evals):
    '''
    Trains the GP classifier using both successful and timed-out evaluations.
    Successful evaluations are labeled as 1 (halted), timed-out as 0 (not halted)
    '''
    if not successful_evals and not timed_out_evals:
        print("Warning: No evaluations yet to train GP classifier")
        return gp_classifier

    X_halted = np.array([s[0] for s in successful_evals])
    X_timed_out = np.array(timed_out_evals)

    X_train = np.concatenate((X_halted, X_timed_out)).reshape(-1,1)
    y_train = np.concatenate((np.ones_like(X_halted), np.zeros_like(X_timed_out)))

    # Check for at least two classes
    if len(np.unique(y_train)) < 2:
        print("Warning: Only one class present in training data for GP classifier. Skipping fit.")
        return gp_classifier

    X_train_scaled = gp_classifier.scaler_.fit_transform(X_train)
    gp_classifier.fit(X_train_scaled, y_train)

    return gp_classifier

# Visualization
def plot_models_and_data(x_domain, successful_evals, timed_out_evals, gp_regressor=None, gp_classifier=None):
    """
    Plots the current state of the AI's models and the observed data.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Extract data for plotting
    x_success = np.array([s[0] for s in successful_evals]).reshape(-1, 1)
    y_success = np.array([s[1] for s in successful_evals])
    x_timed_out = np.array(timed_out_evals).reshape(-1, 1)

    # --- Plot 1: True Objective and GP Regressor's Belief ---
    ax1.set_ylabel('Objective Value', color='tab:blue')
    ax1.plot(x_domain, [true_objective_function(x) for x in x_domain], 
             '--', color='gray', label='True f(x) (Hidden)')
    ax1.scatter(x_success, y_success, color='blue', label='Successful Eval', zorder=5)
    
    if gp_regressor:
        y_pred, sigma = gp_regressor.predict(x_domain.reshape(-1, 1), return_std=True)
        ax1.plot(x_domain, y_pred, color='tab:blue', label='GP Mean Prediction')
        ax1.fill_between(x_domain, y_pred - sigma, y_pred + sigma, alpha=0.2, color='tab:blue', 
                         label='GP Std Dev')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title("AI's Belief about Objective Value (f(x))")


    # --- Plot 2: Halting Probability and GP Classifier's Belief ---
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability', color='tab:red')
    ax2.plot(x_domain, [halting_probability_function(x) for x in x_domain], 
             '--', color='gray', label='True P(halts|x) (Hidden)')
    ax2.scatter(x_timed_out, np.zeros_like(x_timed_out), marker='x', color='red', s=100, linewidth=2,
                label='Timed Out Eval', zorder=5)
    
    # We can plot successful evals at y=1 to show they halted
    ax2.scatter(x_success, np.ones_like(x_success), marker='o', facecolors='none', edgecolors='blue', s=50,
                label='Halted Eval (Y=1)', zorder=5)

    if gp_classifier:
        # GPC predicts probabilities, not mean/std
        # It expects X in a specific format for prediction (e.g., scaled if trained with scaling)
        if hasattr(gp_classifier, 'scaler_') and gp_classifier.scaler_ is not None:
            x_domain_scaled = gp_classifier.scaler_.transform(x_domain.reshape(-1, 1))
        else:
            x_domain_scaled = x_domain.reshape(-1, 1)

        prob_pred = gp_classifier.predict_proba(x_domain_scaled)[:, 1] # Probability of class 1 (halted)
        ax2.plot(x_domain, prob_pred, color='tab:red', label='GP Classifier P(halts)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("AI's Belief about Halting Probability")

    plt.tight_layout()
    plt.show()

# Example
if __name__ == '__main__':
    print("---Testing Phase 2 components: Surrogate Models")

    # Define the domain for our x values for plotting
    x_domain_plot = np.linspace(0, 10, 200)

    # 1. Collect some initial data using the Phase 1 components
    print("\n--- Collecting Initial Data ---")
    initial_successful, initial_timed_out = collect_initial_data(
        num_samples=10, x_min=0.0, x_max=10.0, time_limit_per_eval=0.05, penalty_value=-1000.0
    )

    print("\n--- Training Surrogate Models ---")
    # 2. Initialize and Train the GP Regressor (Mf)
    gp_reg = initialize_gp_regressor()
    gp_reg = train_gp_regressor(gp_reg, initial_successful)
    print(f"GP Regressor trained. Num successful points: {len(initial_successful)}")

    # 3. Initialize and Train the GP Classifier (MH)
    gp_clf = initialize_gp_classifier()
    gp_clf = train_gp_classifier(gp_clf, initial_successful, initial_timed_out)
    print(f"GP Classifier trained. Num total points: {len(initial_successful) + len(initial_timed_out)}")

    # 4. Visualize the initial state of the models
    print("\n--- Visualizing Initial Model Beliefs ---")
    plot_models_and_data(x_domain_plot, initial_successful, initial_timed_out, gp_reg, gp_clf)

    print("\nPhase 2 Complete. Models are trained based on initial data.")