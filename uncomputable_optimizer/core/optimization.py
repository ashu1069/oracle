import numpy as np
import matplotlib.pyplot as plt 
from .simulator import true_objective_function, collect_initial_data, halting_probability_function, uncomputable_oracle
from .brain import initialize_gp_regressor, initialize_gp_classifier, train_gp_regressor, train_gp_classifier, plot_models_and_data
from .decision_maker import calculate_expected_improvement, calculate_risk_aware_acquisition
from .visualization import create_optimization_animation
from typing import List, Tuple, Optional

# Main Optimization Loop
def run_uncomputable_optimizer(
    num_initial_samples=5,
    num_iterations=20,
    x_min=0.0,
    x_max=10.0,
    oracle_time_limit=0.05,
    oracle_penalty_value=-1000.0,
    save_animation_path: Optional[str] = None,
    animation_fps: int = 2,
    animation_dpi: int = 100,
    objective_function=None,
    halting_probability_function=None
):
    """
    Runs the full Uncomputable Oracle Optimizer.

    Args:
        num_initial_samples (int): Number of random samples to start with.
        num_iterations (int): Number of optimization iterations after initial samples.
        x_min (float): Minimum value for x in the domain.
        x_max (float): Maximum value for x in the domain.
        oracle_time_limit (float): Time limit for each oracle evaluation.
        oracle_penalty_value (float): Penalty returned by oracle on timeout.
        save_animation_path (str, optional): Path to save the optimization animation.
        animation_fps (int): Frames per second for the animation.
        animation_dpi (int): Dots per inch for the saved animation.
        objective_function (callable, optional): Custom objective function.
        halting_probability_function (callable, optional): Custom halting probability function.
    """
    print("--- Starting Uncomputable Oracle Optimizer ---")
    print(f"Domain: [{x_min}, {x_max}] | Initial Samples: {num_initial_samples} | Iterations: {num_iterations}")
    print(f"Oracle Eval Time Limit: {oracle_time_limit}s | Timeout Penalty: {oracle_penalty_value}")

    # Initialize data storage
    successful_evals = [] # (x, f_val)
    timed_out_evals = []  # x
    best_f_so_far = -np.inf
    best_x_so_far = None
    
    # Store history for animation
    optimization_history = [] # (iteration, successful_evals, timed_out_evals, gp_regressor, gp_classifier, acquisition_vals, best_x_to_probe, best_f_so_far)

    # Define candidate points for acquisition function evaluation
    x_candidates_for_acquisition = np.linspace(x_min, x_max, 200)
    x_domain_plot = x_candidates_for_acquisition  # For visualization

    # 1. Collect initial data
    initial_successful, initial_timed_out = collect_initial_data(
        num_samples=num_initial_samples, x_min=x_min, x_max=x_max,
        time_limit_per_eval=oracle_time_limit, penalty_value=oracle_penalty_value,
        objective_function=objective_function, halting_probability_function=halting_probability_function
    )
    successful_evals.extend(initial_successful)
    timed_out_evals.extend(initial_timed_out)

    # Update best_f_so_far from initial samples
    if successful_evals:
        best_f_so_far = max([s[1] for s in successful_evals])
        best_x_so_far = [s[0] for s in successful_evals if s[1] == best_f_so_far][0]

    # Initialize GP models
    gp_regressor = initialize_gp_regressor()
    gp_classifier = initialize_gp_classifier()

    # Train models for initial state
    gp_regressor = train_gp_regressor(gp_regressor, successful_evals)
    gp_classifier = train_gp_classifier(gp_classifier, successful_evals, timed_out_evals)
    
    # Calculate initial acquisition values
    initial_acq_vals = calculate_risk_aware_acquisition(
        x_candidates_for_acquisition, gp_regressor, gp_classifier, best_f_so_far
    )
    best_x_for_initial_plot = x_candidates_for_acquisition[np.argmax(initial_acq_vals)] \
                              if len(initial_successful) > 0 else (x_min + x_max) / 2

    # Compute initial GP predictions for history
    if len(successful_evals) > 0:
        obj_mean, obj_std = gp_regressor.predict(x_candidates_for_acquisition.reshape(-1, 1), return_std=True)
    else:
        obj_mean = np.zeros_like(x_candidates_for_acquisition)
        obj_std = np.zeros_like(x_candidates_for_acquisition)
    if hasattr(gp_classifier, 'classes_') and len(getattr(gp_classifier, 'classes_', [])) > 1:
        x_scaled = gp_classifier.scaler_.transform(x_candidates_for_acquisition.reshape(-1, 1))
        halt_mean = gp_classifier.predict_proba(x_scaled)[:, 1]
        halt_std = np.sqrt(halt_mean * (1 - halt_mean))
    else:
        halt_mean = np.full_like(x_candidates_for_acquisition, 0.5)
        halt_std = np.full_like(x_candidates_for_acquisition, 0.5)
    successful_vals = [s[1] for s in successful_evals]

    # Store initial state in history
    optimization_history.append({
        'iteration': 0,
        'successful_evals': successful_evals.copy(),
        'timed_out_evals': timed_out_evals.copy(),
        'gp_regressor': gp_regressor,
        'gp_classifier': gp_classifier,
        'acquisition_vals': initial_acq_vals,
        'acquisition_values': initial_acq_vals,  # alias for animation
        'best_x_to_probe': best_x_for_initial_plot,
        'current_probe_x': best_x_for_initial_plot,
        'best_x_so_far': best_x_so_far,
        'best_f_so_far': best_f_so_far,
        'obj_mean': obj_mean,
        'obj_std': obj_std,
        'gp_mean': obj_mean,
        'gp_std': obj_std,
        'halt_mean': halt_mean,
        'halt_std': halt_std,
        'halting_gp_mean': halt_mean,
        'halting_gp_std': halt_std,
        'successful_vals': successful_vals,
        'successful_x': [s[0] for s in successful_evals],
        'successful_y': [s[1] for s in successful_evals],
        'timeout_x': timed_out_evals,
    })

    # 2. Main Optimization Loop
    for i in range(1, num_iterations + 1):
        print(f"\n--- Optimization Iteration {i}/{num_iterations} ---")

        # a. Train Models with all current data
        gp_regressor = train_gp_regressor(gp_regressor, successful_evals)
        gp_classifier = train_gp_classifier(gp_classifier, successful_evals, timed_out_evals)

        # b. Calculate Acquisition Function for candidates
        acquisition_vals = calculate_risk_aware_acquisition(
            x_candidates_for_acquisition, gp_regressor, gp_classifier, best_f_so_far
        )

        # c. Select next point to probe (maximize acquisition function)
        best_x_to_probe = x_candidates_for_acquisition[np.argmax(acquisition_vals)]
        print(f"AI recommends probing x = {best_x_to_probe:.2f}")

        # d. Evaluate the chosen point using the oracle
        print(f"  Calling Oracle for x = {best_x_to_probe:.2f}...", end="")
        value, halted = uncomputable_oracle(
            best_x_to_probe, time_limit_per_eval=oracle_time_limit,
            penalty_value=oracle_penalty_value,
            objective_function=objective_function,
            halting_probability_function=halting_probability_function
        )

        # e. Update data storage
        if halted:
            successful_evals.append((best_x_to_probe, value))
            print(f" Halted! Value: {value:.2f}")
            if value > best_f_so_far:
                best_f_so_far = value
                best_x_so_far = best_x_to_probe
                print(f"  New best found: f(x) = {best_f_so_far:.2f} at x = {best_x_so_far:.2f}")
        else:
            timed_out_evals.append(best_x_to_probe)
            print(f" Timed out! Penalty: {value:.2f}")

        # Compute GP predictions for history
        if len(successful_evals) > 0:
            obj_mean, obj_std = gp_regressor.predict(x_candidates_for_acquisition.reshape(-1, 1), return_std=True)
        else:
            obj_mean = np.zeros_like(x_candidates_for_acquisition)
            obj_std = np.zeros_like(x_candidates_for_acquisition)
        if hasattr(gp_classifier, 'classes_') and len(getattr(gp_classifier, 'classes_', [])) > 1:
            x_scaled = gp_classifier.scaler_.transform(x_candidates_for_acquisition.reshape(-1, 1))
            halt_mean = gp_classifier.predict_proba(x_scaled)[:, 1]
            halt_std = np.sqrt(halt_mean * (1 - halt_mean))
        else:
            halt_mean = np.full_like(x_candidates_for_acquisition, 0.5)
            halt_std = np.full_like(x_candidates_for_acquisition, 0.5)
        successful_vals = [s[1] for s in successful_evals]

        # Store current state in history
        optimization_history.append({
            'iteration': i,
            'successful_evals': successful_evals.copy(),
            'timed_out_evals': timed_out_evals.copy(),
            'gp_regressor': gp_regressor,
            'gp_classifier': gp_classifier,
            'acquisition_vals': acquisition_vals,
            'acquisition_values': acquisition_vals,  # alias for animation
            'best_x_to_probe': best_x_to_probe,
            'current_probe_x': best_x_to_probe,
            'best_x_so_far': best_x_so_far,
            'best_f_so_far': best_f_so_far,
            'obj_mean': obj_mean,
            'obj_std': obj_std,
            'gp_mean': obj_mean,
            'gp_std': obj_std,
            'halt_mean': halt_mean,
            'halt_std': halt_std,
            'halting_gp_mean': halt_mean,
            'halting_gp_std': halt_std,
            'successful_vals': successful_vals,
            'successful_x': [s[0] for s in successful_evals],
            'successful_y': [s[1] for s in successful_evals],
            'timeout_x': timed_out_evals,
        })

    print("\n--- Optimization Finished ---")
    print(f"Final Best Found: f(x) = {best_f_so_far:.2f} at x = {best_x_so_far:.2f}")
    print(f"Total Evaluations: {len(successful_evals) + len(timed_out_evals)}")
    print(f"Successful Evaluations: {len(successful_evals)}")
    print(f"Timed Out Evaluations: {len(timed_out_evals)}")
    
    # Create and save animation if requested
    if save_animation_path:
        print(f"\nCreating optimization animation...")
        create_optimization_animation(
            optimization_history,
            x_domain_plot,
            save_path=save_animation_path,
            fps=animation_fps,
            dpi=animation_dpi
        )
        print(f"Animation saved to: {save_animation_path}")
    
    return best_x_so_far, best_f_so_far, optimization_history

def plot_optimization_progress(x_domain, successful_evals, timed_out_evals,
                               gp_regressor, gp_classifier, acquisition_vals,
                               best_x_to_probe, f_best_so_far, iteration):
    """
    Combines all plots to show the AI's progress in one comprehensive view.
    """
    # Create figure with a larger size to accommodate explanations
    fig = plt.figure(figsize=(16, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[0.1, 1, 1, 1], hspace=0.3)
    
    # Add title with iteration info and best value
    fig.suptitle(f'Optimization Progress - Iteration {iteration}\nBest Value Found: {f_best_so_far:.2f}', 
                 fontsize=16, y=0.98)

    # Extract data for plotting
    x_success = np.array([s[0] for s in successful_evals]).reshape(-1, 1)
    y_success = np.array([s[1] for s in successful_evals])
    x_timed_out = np.array(timed_out_evals).reshape(-1, 1)

    # Ax1: True Objective, GP Regressor's Belief, and Sampled Points 
    ax1 = fig.add_subplot(gs[1])
    ax1.set_ylabel('Objective Value', fontsize=12)
    
    # Plot true objective function
    ax1.plot(x_domain, [true_objective_function(x) for x in x_domain], 
             '--', color='gray', label='True f(x) (Hidden)')
    
    # Plot GP mean and std dev if regressor is trained
    if gp_regressor is not None and len(gp_regressor.X_train_) > 0:
        y_pred, sigma = gp_regressor.predict(x_domain.reshape(-1, 1), return_std=True)
        ax1.plot(x_domain, y_pred, color='tab:blue', label='GP Mean Prediction')
        ax1.fill_between(x_domain, y_pred - sigma, y_pred + sigma, alpha=0.2, color='tab:blue', 
                         label='GP Uncertainty')

    # Plot observed points
    ax1.scatter(x_success, y_success, color='blue', s=80, edgecolors='k', label='Successful Eval', zorder=5)
    ax1.scatter(x_timed_out, np.full_like(x_timed_out, -900), marker='x', color='red', s=100, linewidth=2,
                label='Timed Out Eval (penalty)', zorder=5)
    
    ax1.axhline(f_best_so_far, color='green', linestyle=':', label=f'Best f(x) so far: {f_best_so_far:.2f}')
    
    # Set y-axis range for objective value plot
    y_min = min(-1000, min(y_success) if len(y_success) > 0 else -20)
    y_max = max(20, max(y_success) if len(y_success) > 0 else 20)
    ax1.set_ylim(y_min, y_max)
    
    # Add explanation text box
    ax1.text(0.02, 0.98, 
             "What you're seeing:\n"
             "• Blue line: AI's current best guess of f(x)\n"
             "• Shaded area: AI's uncertainty\n"
             "• Blue dots: Successful evaluations\n"
             "• Red X's: Timed out evaluations\n"
             "• Green line: Best value found so far",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title("AI's Belief about Objective Value (f(x)) and Observations", fontsize=14, pad=20)

    # Ax2: Halting Probability and GP Classifier's Belief 
    ax2 = fig.add_subplot(gs[2])
    ax2.set_ylabel('P(halts|x)', fontsize=12)
    
    # Plot true halting probability
    ax2.plot(x_domain, [halting_probability_function(x) for x in x_domain], 
             '--', color='gray', label='True P(halts|x) (Hidden)')
    
    # Plot GP Classifier's prediction if trained
    if gp_classifier is not None and hasattr(gp_classifier, 'classes_') and len(gp_classifier.classes_) > 1:
        x_domain_scaled_clf = gp_classifier.scaler_.transform(x_domain.reshape(-1, 1))
        prob_pred = gp_classifier.predict_proba(x_domain_scaled_clf)[:, 1]
        ax2.plot(x_domain, prob_pred, color='tab:red', label='GP Classifier P(halts)')
    
    # Plot halted (y=1) and timed out (y=0) points
    ax2.scatter(x_timed_out, np.zeros_like(x_timed_out), marker='x', color='red', s=100, linewidth=2,
                label='Timed Out Eval (Y=0)', zorder=5)
    ax2.scatter(x_success, np.ones_like(x_success), marker='o', facecolors='none', edgecolors='blue', s=50,
                label='Halted Eval (Y=1)', zorder=5)
    
    # Set y-axis range for halting probability plot
    ax2.set_ylim(-0.05, 1.05)
    
    # Add explanation text box
    ax2.text(0.02, 0.98,
             "What you're seeing:\n"
             "• Red line: AI's belief about halting probability\n"
             "• Blue circles: Points that halted (Y=1)\n"
             "• Red X's: Points that timed out (Y=0)\n"
             "• Higher probability = safer to evaluate",
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("AI's Belief about Halting Probability (P(halts|x))", fontsize=14, pad=20)

    # Ax3: Risk-Aware Acquisition Function 
    ax3 = fig.add_subplot(gs[3])
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Acquisition Value', fontsize=12)
    
    # Plot acquisition function
    ax3.plot(x_domain, acquisition_vals, color='purple', label='Risk-Aware Acquisition Function')
    
    # Highlight the next probe point
    ax3.axvline(best_x_to_probe, color='green', linestyle=':', label=f'Next Probe x: {best_x_to_probe:.2f}')
    ax3.scatter(best_x_to_probe, np.max(acquisition_vals), color='green', marker='*', s=200, zorder=5)
    
    # Set y-axis range for acquisition function plot
    acq_min = min(0, np.min(acquisition_vals))
    acq_max = max(0.1, np.max(acquisition_vals))
    ax3.set_ylim(acq_min, acq_max)
    
    # Add explanation text box
    ax3.text(0.02, 0.98,
             "What you're seeing:\n"
             "• Purple line: AI's decision function\n"
             "• Green star: Next point to evaluate\n"
             "• Higher value = more promising point\n"
             "• Combines expected improvement and halting probability",
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_title("AI's Decision: Risk-Aware Acquisition Function", fontsize=14, pad=20)

    # Add overall explanation at the bottom
    fig.text(0.5, 0.02,
             "The AI is learning to optimize while avoiding computational timeouts.\n"
             "It balances finding high values (top plot) with avoiding timeouts (middle plot)\n"
             "to make smart decisions about where to probe next (bottom plot).",
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig

if __name__ == "__main__":
    # Define the domain for plotting (needs to be consistent)
    x_domain_plot = np.linspace(0, 10, 200) 
    
    run_uncomputable_optimizer(
        num_initial_samples=5,
        num_iterations=25, # Adjust number of iterations to see more learning
        x_min=0.0,
        x_max=10.0,
        oracle_time_limit=0.05, # Shorter time limit makes it faster to run but riskier
        oracle_penalty_value=-1000.0
    )