import numpy as np
import time
import random
import matplotlib.pyplot as plt 

# The Hidden Truth
def true_objective_function(x):
    '''
    The true underlying function we want to optimize -- unknown to our AI optimizer.
    Our goal is to find the x that maximizes this value.
    '''
    return -((x-5)**2) + 10

def halting_probability_function(x):
    '''
    Determines the probability that the "uncomputable oracle" will halt for a given input x.
    This introduces uncomputable zones -- also unknown to our AI optimizer

    Design choices:
        - high probability of hanging (low halt_prob) near x = 2 and x=8
        - high probability of halting (high halt_prob) near the optimum x=5
        - moderate probability elsewhere
    '''
    normalized_x = x/10.0

    # Let's use a combination of sine waves and constants to create interesting zones
    # This function is chosen to be somewhat non-trivial for the AI to learn

    prob = 0.5 + 0.4 * np.sin(normalized_x * 2 * np.pi) + 0.1 * np.cos(normalized_x * 5 * np.pi)

    # Zones
    if 1.8 < x < 2.2 or 7.8 < x < 8.2:
        prob = 0.1
    elif 4.8 < x < 5.2:
        prob = min (prob, 0.95)

    return np.clip(prob, 0.05, 0.95)

# The Uncomputable Simulator
def uncomputable_oracle(x, time_limit_per_eval = 0.05, penalty_value = -1000.0):
    '''
    This function simulates the black-box objective evaluation; takes an input "x" and attempts to compute "true_objective_function(x)".
    However, based on "halting_probability_function(x)", it might "hang" and return a penalty instead.
    
    Args:
        x (float): input value for the objective function
        time_limit_per_eval (float): the maximum time allowed for a single evaluation
        penalty_value (float): the value returned if the computation "times out"
    Returns:
        float: the true objective if the computation halts, or penalty_value
        bool: True if halted, false if it times out
    '''
    start_time = time.time()
    
    # Determine if this specific evaluation "halts" based on the hidden probability
    halt_chance = halting_probability_function(x)

    if random.random() < halt_chance:
        # simulate computation time for a halted evaluation; this makes it feel like a real computation, even if quick
        simulated_computation_time = random.uniform(0.001, time_limit_per_eval * 0.5)
        time.sleep(simulated_computation_time)

        # Ensure we don't exceed the time limit accidentally, though unlikely here
        if (time.time() - start_time) > time_limit_per_eval:
            # this is for rare edge case if simulated_computation_time is too high
            print(f"Warning: Halted computation for x = {x:.2f} exceeded time limit. Timeout!")
            return penalty_value, False 
        
        return true_objective_function(x), True

    else:
        return penalty_value, False

# Initial Data Collection Strategy

def collect_initial_data(num_samples = 5, x_min = 0.0, x_max = 10.0, **oracle_kwargs):
    '''
    Collects an intial set of data points by randomly sampling the domain.
    This provides the starting observations for our AI's models
    '''
    successful_evals = [] # stores (x, f_val) for halted computations
    timed_out_evals = [] # stores x for timed-out computations

    print(f"Collecting initial {num_samples} data points...")
    for i in range(num_samples):
        x = random.uniform(x_min, x_max)
        print(f"Probing x = {x:0.2f} (sample {i+1}/{num_samples})...", end="")

        value, halted = uncomputable_oracle(x, **oracle_kwargs)

        if halted:
            successful_evals.append((x, value))
            print(f"Halted! Value: {value: 0.2f}")
        else:
            timed_out_evals.append(x)
            print(f"Timed out! Penalty: {value:0.2f}")

    return successful_evals, timed_out_evals

# Visualizing the hidden truth -- for our understanding, not AI's

def plot_hidden_truth(x_domain):
    true_f_vals = [true_objective_function(x) for x in x_domain]
    halt_prob_vals = [halting_probability_function(x) for x in x_domain]

    fig, ax1 = plt.subplots(figsize = (10,6))
    color = "tab:blue"
    ax1.set_xlabel('x')
    ax1.set_ylabel('True Objective Function, f(x)', color=color)
    ax1.plot(x_domain, true_f_vals, color=color, linestyle = '--', label='True f(x)')
    ax1.tick_params(axis='y', labelcolor = color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Halting Probability P(halts|x)', color = color)
    ax2.plot(x_domain, halt_prob_vals, color=color, label='P(halts|x)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.title('Hidden Truth: True Objective and Halting Probability')
    fig.legend(loc = "upper left", bbox_to_anchor = (0.1, 0.9))
    plt.show()

# Example Usage (to test the oracle simulator)
if __name__ == "__main__":
    print("---Testing Phase 1 Components---")

    # Define the domain for our x values
    x_domain = np.linspace(0,10,100)

    # Plot the hidden truth for our reference
    plot_hidden_truth(x_domain)

    # Test the oracle with a few specific points
    print(f"\n---Probing specific points with the Oracle---")
    test_points = [1.0, 2.0, 5.0, 8.0, 9.0]

    for x_val in test_points:
        print(f"Probing x = {x_val:0.2f}...", end="")
        value, halted = uncomputable_oracle(x_val, time_limit_per_eval=0.1)
        if halted:
            print(f"Halted! f({x_val:0.2f}) = {value:0.2f}")
        else:
            print(f"Timed out! Penalty: {value:0.2f}")

    # Collecting some initial data
    initial_successful, initial_timed_out = collect_initial_data(num_samples=7, x_min = 0.0, x_max = 10.0, time_limit_per_eval=0.1, penalty_value=-1000.0)

    print("\n ---Initial Data Collected---")
    print(f"Successful Evaluations: {initial_successful}")
    print(f"Timed-out Evaluations: {initial_timed_out}")

