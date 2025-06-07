import os
import numpy as np
from uncomputable_optimizer.core.optimization import run_uncomputable_optimizer
from uncomputable_optimizer.core.visualization import create_interactive_visualization
import subprocess
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from uncomputable_optimizer.core.simulator import true_objective_function, halting_probability_function

def create_separate_plot_animation(history, x_domain, output_dir, fps=2, dpi=150, video_format='mp4'):
    """Create separate animations for each plot type."""
    # Use the same x grid as used in the optimization/acquisition
    if len(history) > 0 and 'acquisition_vals' in history[0]:
        x_grid = np.linspace(x_domain[0], x_domain[1], len(history[0]['acquisition_vals']))
    else:
        x_grid = np.linspace(x_domain[0], x_domain[1], 200)

    def update(frame):
        successful_evals = history[frame]['successful_evals']
        timed_out_evals = history[frame]['timed_out_evals']
        gp_regressor = history[frame]['gp_regressor']
        gp_classifier = history[frame]['gp_classifier']
        acquisition_vals = history[frame]['acquisition_vals']
        current_probe_x = history[frame]['current_probe_x']
        best_x_so_far = history[frame]['best_x_so_far']
        best_f_so_far = history[frame]['best_f_so_far']
        
        for ax in axes:
            ax.clear()
        
        # Plot acquisition function
        axes[0].plot(x_grid, acquisition_vals, 'b-', label='Acquisition')
        axes[0].set_title('Risk-Aware Acquisition Function')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Acquisition Value')
        axes[0].grid(True)
        axes[0].legend()
        if current_probe_x is not None:
            axes[0].axvline(x=current_probe_x, color='orange', linestyle='--', alpha=0.5)
            axes[0].scatter([current_probe_x], [0], color='orange', marker='*', s=200, label='Current Probe')
        
        # Plot halting probability
        if gp_classifier is not None:
            if hasattr(gp_classifier, 'scaler_') and gp_classifier.scaler_ is not None:
                x_grid_scaled = gp_classifier.scaler_.transform(x_grid.reshape(-1, 1))
            else:
                x_grid_scaled = x_grid.reshape(-1, 1)
            halt_probs = gp_classifier.predict_proba(x_grid_scaled)[:, 1]
            axes[1].plot(x_grid, halt_probs, 'g-', label='Halting Probability')
            axes[1].set_title('Halting Probability')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('P(Halts)')
            axes[1].grid(True)
            axes[1].legend()
            if current_probe_x is not None:
                axes[1].axvline(x=current_probe_x, color='orange', linestyle='--', alpha=0.5)
                axes[1].scatter([current_probe_x], [0], color='orange', marker='*', s=200, label='Current Probe')
        
        # Plot objective function
        if gp_regressor is not None:
            mean, std = gp_regressor.predict(x_grid.reshape(-1, 1), return_std=True)
            axes[2].plot(x_grid, mean, 'r-', label='GP Mean')
            axes[2].fill_between(x_grid, mean - 2*std, mean + 2*std, color='r', alpha=0.2, label='95% Confidence')
            true_y = -(x_grid - 5)**2 + 10
            axes[2].plot(x_grid, true_y, 'k--', label='True Objective', alpha=0.5)
            if len(successful_evals) > 0:
                x_success = np.array([x for x, _ in successful_evals])
                y_success = np.array([y for _, y in successful_evals])
                axes[2].scatter(x_success, y_success, color='g', marker='o', label='Successful Evals')
            if len(timed_out_evals) > 0:
                x_timeout = np.array(timed_out_evals)
                y_timeout = np.full_like(x_timeout, -1000.0)
                axes[2].scatter(x_timeout, y_timeout, color='r', marker='x', label='Timed Out Evals')
            if current_probe_x is not None:
                axes[2].axvline(x=current_probe_x, color='orange', linestyle='--', alpha=0.5)
                axes[2].scatter([current_probe_x], [0], color='orange', marker='*', s=200, label='Current Probe')
            if best_x_so_far is not None and best_f_so_far is not None:
                axes[2].axvline(x=best_x_so_far, color='green', linestyle='--', alpha=0.5)
                axes[2].axhline(y=best_f_so_far, color='green', linestyle='--', alpha=0.5)
                axes[2].scatter([best_x_so_far], [best_f_so_far], color='green', marker='D', s=200, label='Best Found')
            axes[2].set_title('Objective Function')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('f(x)')
            axes[2].grid(True)
            axes[2].legend()
        fig.suptitle(f'Iteration {frame}', fontsize=16)
        info_text = f'Current Probe: x = {current_probe_x:.2f}\nBest Found: x = {best_x_so_far:.2f}, f(x) = {best_f_so_far:.2f}'
        fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=3.0)
    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps)
    for i, title in enumerate(['acquisition', 'halting', 'objective']):
        anim.save(f'{output_dir}/{title}_animation.{video_format}', 
                 fps=fps, dpi=dpi, writer='ffmpeg')
    plt.close(fig)

# New function for synchronized side-by-side animation
def create_synchronized_side_by_side_animation(history, x_domain, output_dir, fps, dpi, video_format):
    """
    Creates a synchronized side-by-side animation of acquisition, halting, and objective plots.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Synchronized Optimization Progress', fontsize=16)

    # Define candidate points for acquisition function evaluation
    x_candidates = np.linspace(x_domain[0], x_domain[-1], 200)

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Extract data for current frame
        current_state = history[frame]
        successful_evals = current_state['successful_evals']
        timed_out_evals = current_state['timed_out_evals']
        gp_regressor = current_state['gp_regressor']
        gp_classifier = current_state['gp_classifier']
        acquisition_vals = current_state['acquisition_values']
        current_probe_x = current_state['current_probe_x']
        best_x_so_far = current_state['best_x_so_far']
        best_f_so_far = current_state['best_f_so_far']

        # Plot acquisition function
        ax1.plot(x_candidates, acquisition_vals, color='purple', label='Acquisition Function')
        ax1.axvline(current_probe_x, color='orange', linestyle='--', label='Current Probe')
        ax1.scatter(current_probe_x, np.max(acquisition_vals), color='orange', marker='*', s=200, zorder=5)
        ax1.set_title('Acquisition Function')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Acquisition Value')
        ax1.legend()

        # Plot halting probability
        ax2.plot(x_domain, [halting_probability_function(x) for x in x_domain], '--', color='gray', label='True P(halts|x) (Hidden)')
        if gp_classifier is not None and hasattr(gp_classifier, 'classes_') and len(gp_classifier.classes_) > 1:
            x_domain_scaled = gp_classifier.scaler_.transform(x_domain.reshape(-1, 1))
            prob_pred = gp_classifier.predict_proba(x_domain_scaled)[:, 1]
            ax2.plot(x_domain, prob_pred, color='tab:red', label='GP Classifier P(halts)')
        ax2.scatter(timed_out_evals, np.zeros_like(timed_out_evals), marker='x', color='red', s=100, linewidth=2, label='Timed Out Eval (Y=0)', zorder=5)
        ax2.scatter([s[0] for s in successful_evals], np.ones_like([s[0] for s in successful_evals]), marker='o', facecolors='none', edgecolors='blue', s=50, label='Halted Eval (Y=1)', zorder=5)
        ax2.axvline(current_probe_x, color='orange', linestyle='--', label='Current Probe')
        ax2.set_title('Halting Probability')
        ax2.set_xlabel('x')
        ax2.set_ylabel('P(halts|x)')
        ax2.legend()

        # Plot objective function
        ax3.plot(x_domain, [true_objective_function(x) for x in x_domain], '--', color='gray', label='True f(x) (Hidden)')
        if gp_regressor is not None and len(gp_regressor.X_train_) > 0:
            y_pred, sigma = gp_regressor.predict(x_domain.reshape(-1, 1), return_std=True)
            ax3.plot(x_domain, y_pred, color='tab:blue', label='GP Mean Prediction')
            ax3.fill_between(x_domain, y_pred - sigma, y_pred + sigma, alpha=0.2, color='tab:blue', label='GP Uncertainty')
        ax3.scatter([s[0] for s in successful_evals], [s[1] for s in successful_evals], color='blue', s=80, edgecolors='k', label='Successful Eval', zorder=5)
        ax3.scatter(timed_out_evals, np.full_like(timed_out_evals, -900), marker='x', color='red', s=100, linewidth=2, label='Timed Out Eval (penalty)', zorder=5)
        ax3.axhline(best_f_so_far, color='green', linestyle=':', label=f'Best f(x) so far: {best_f_so_far:.2f}')
        ax3.axvline(current_probe_x, color='orange', linestyle='--', label='Current Probe')
        ax3.set_title('Objective Function')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Objective Value')
        ax3.legend()

        # Add objective function equation and details
        objective_eq = r"$f(x) = -(x-5)^2 + 10$"
        objective_details = (
            "Objective Function:\n"
            "• Quadratic function centered at x=5\n"
            "• Maximum value of 10 at x=5\n"
            "• Domain: [0, 10]\n"
            "• Some points may timeout during evaluation"
        )
        ax3.text(0.02, 0.98, objective_eq, transform=ax3.transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.text(0.02, 0.85, objective_details, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps, blit=False)
    anim.save(f'{output_dir}/synchronized_animation.{video_format}', fps=fps, dpi=dpi)
    plt.close(fig)

def create_demo_video(
    output_dir: str = "demo_video",
    num_iterations: int = 30,
    fps: int = 2,
    dpi: int = 150,
    video_format: str = "mp4"
):
    """
    Create high-quality demo videos of the optimization process.
    
    Args:
        output_dir: Directory to save videos and intermediate files
        num_iterations: Number of optimization iterations
        fps: Frames per second for the videos
        dpi: Resolution for the frames
        video_format: Output video format (mp4 or gif)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization
    print("Running optimization...")
    best_x, best_f, history = run_uncomputable_optimizer(
        num_initial_samples=5,
        num_iterations=num_iterations,
        x_min=0.0,
        x_max=10.0,
        oracle_time_limit=0.05,
        oracle_penalty_value=-1000.0
    )
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    x_domain = np.linspace(0, 10, 200)
    interactive_fig = create_interactive_visualization(
        history,
        x_domain,
        save_path=os.path.join(output_dir, "interactive_visualization.html")
    )
    
    # Create separate animations for each plot
    print("Creating separate plot animations...")
    create_separate_plot_animation(
        history=history,
        x_domain=x_domain,
        output_dir=output_dir,
        fps=fps,
        dpi=dpi,
        video_format=video_format
    )
    
    print("Creating synchronized side-by-side animation...")
    create_synchronized_side_by_side_animation(history, x_domain, output_dir, fps, dpi, video_format)
    
    print(f"Demo videos created successfully in {output_dir}/")
    print(f"Interactive visualization saved as {output_dir}/interactive_visualization.html")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create demo videos for the Uncomputable Optimizer')
    parser.add_argument('--output_dir', type=str, default='demo_video',
                      help='Directory to save the demo videos')
    parser.add_argument('--num_iterations', type=int, default=30,
                      help='Number of optimization iterations')
    parser.add_argument('--fps', type=int, default=2,
                      help='Frames per second for the animations')
    parser.add_argument('--dpi', type=int, default=150,
                      help='DPI for the animations')
    parser.add_argument('--video_format', type=str, default='mp4',
                      help='Video format (mp4 or gif)')
    
    args = parser.parse_args()
    
    create_demo_video(
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        fps=args.fps,
        dpi=args.dpi,
        video_format=args.video_format
    ) 