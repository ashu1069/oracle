import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Dict
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from .simulator import true_objective_function, halting_probability_function
from matplotlib import animation
import os
matplotlib.use('Agg')  # Use non-interactive backend for saving animations

class OptimizationAnimator:
    """
    Creates animated visualizations of the optimization process.
    """
    def __init__(self, x_grid, oracle_penalty_value):
        """Initialize the animator with the grid points and penalty value."""
        self.x_grid = x_grid
        self.oracle_penalty_value = oracle_penalty_value
        self.history = None
        
        # Create figure and axes
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self._init_plots()
        
    def _init_plots(self):
        """Initialize all plot elements."""
        # Objective function plot
        self.obj_line, = self.ax1.plot([], [], 'b-', label='Mean')
        self.obj_std, = self.ax1.plot([], [], 'b--', alpha=0.3, label='±2σ')
        self.success_scatter = self.ax1.scatter([], [], c='g', marker='o', label='Successful')
        self.timeout_scatter = self.ax1.scatter([], [], c='r', marker='x', label='Timed Out')
        self.ax1.set_title('Objective Function')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('f(x)')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Halting probability plot
        self.halt_line, = self.ax2.plot([], [], 'g-', label='Mean')
        self.halt_std, = self.ax2.plot([], [], 'g--', alpha=0.3, label='±2σ')
        self.ax2.set_title('Halting Probability')
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('p(halt)')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Acquisition function plot
        self.acq_line, = self.ax3.plot([], [], 'r-', label='Acquisition')
        self.next_probe_line = self.ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5, visible=False)
        self.next_probe_scatter = self.ax3.scatter([], [], c='k', marker='*', s=100, visible=False)
        self.ax3.set_title('Acquisition Function')
        self.ax3.set_xlabel('x')
        self.ax3.set_ylabel('α(x)')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Iteration counter
        self.iter_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)
        
        # Set x-axis limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(self.x_grid[0], self.x_grid[-1])
        
        plt.tight_layout()
    
    def update(self, frame):
        """Update the plots for each frame of the animation."""
        state = self.history[frame]
        
        # Update objective function plot
        self.obj_line.set_data(self.x_grid, state['obj_mean'])
        self.obj_std.set_data(self.x_grid, state['obj_std'])
        
        # Update halting probability plot
        self.halt_line.set_data(self.x_grid, state['halt_mean'])
        self.halt_std.set_data(self.x_grid, state['halt_std'])
        
        # Update acquisition function plot
        self.acq_line.set_data(self.x_grid, state['acquisition_vals'])
        
        # Update data points
        if state['successful_evals']:
            self.success_scatter.set_offsets(np.column_stack([state['successful_evals'], state['successful_vals']]))
        else:
            self.success_scatter.set_offsets(np.empty((0, 2)))
            
        if state['timed_out_evals']:
            self.timeout_scatter.set_offsets(np.column_stack([state['timed_out_evals'], 
                                                            [self.oracle_penalty_value] * len(state['timed_out_evals'])]))
        else:
            self.timeout_scatter.set_offsets(np.empty((0, 2)))
        
        # Update next probe point
        if state['acquisition_vals'] is not None and len(state['acquisition_vals']) > 0:
            best_x = state['best_x_to_probe']
            ymin, ymax = self.ax3.get_ylim()
            self.next_probe_line.set_xdata([best_x])
            self.next_probe_line.set_ydata([ymin, ymax])
            self.next_probe_line.set_visible(True)
            
            acq_idx = np.argmin(np.abs(self.x_grid - best_x))
            if acq_idx < len(state['acquisition_vals']):
                self.next_probe_scatter.set_offsets([[best_x, state['acquisition_vals'][acq_idx]]])
                self.next_probe_scatter.set_visible(True)
        else:
            self.next_probe_line.set_visible(False)
            self.next_probe_scatter.set_visible(False)
        
        # Update iteration counter
        self.iter_text.set_text(f'Iteration: {frame}')
        
        return (self.obj_line, self.obj_std, self.halt_line, self.halt_std,
                self.acq_line, self.success_scatter, self.timeout_scatter,
                self.next_probe_line, self.next_probe_scatter, self.iter_text)
    
    def create_animation(self, optimization_history, save_path=None, fps=2, dpi=100):
        """Create and save the animation."""
        self.history = optimization_history
        
        anim = animation.FuncAnimation(
            self.fig, self.update,
            frames=len(optimization_history),
            interval=1000/fps,
            blit=True
        )
        
        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
            elif ext == '.gif':
                anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
            else:
                anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        
        return anim

def create_optimization_animation(optimization_history: List[Tuple], x_domain: np.ndarray,
                                save_path: Optional[str] = None, fps: int = 2, dpi: int = 100):
    """
    Convenience function to create an optimization animation.
    
    Args:
        optimization_history: List of tuples containing optimization state at each iteration
        x_domain: Domain points for plotting
        save_path: Path to save the animation (if None, animation will be displayed)
        fps: Frames per second
        dpi: Dots per inch for the saved animation
    """
    animator = OptimizationAnimator(x_domain, 0)
    return animator.create_animation(optimization_history, save_path, fps, dpi)

class InteractiveOptimizationVisualizer:
    """
    Creates interactive visualizations of the optimization process using Plotly.
    """
    def __init__(self, x_domain: np.ndarray):
        self.x_domain = x_domain
        self.fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                "Optimization Progress",
                "AI's Belief about Objective Value (f(x)) and Observations",
                "AI's Belief about Halting Probability (P(halts|x))",
                "AI's Decision: Risk-Aware Acquisition Function"
            ),
            vertical_spacing=0.1,
            shared_xaxes=True,
            specs=[[{"type": "table"}],
                  [{"type": "scatter"}],
                  [{"type": "scatter"}],
                  [{"type": "scatter"}]]
        )
        
    def create_interactive_plot(self, frame_data: Dict):
        """Create an interactive plot for a single frame of the optimization process."""
        # Extract data from dictionary
        iteration = frame_data['iteration']
        successful_evals = frame_data['successful_evals']
        timed_out_evals = frame_data['timed_out_evals']
        gp_regressor = frame_data['gp_regressor']
        gp_classifier = frame_data['gp_classifier']
        acquisition_vals = frame_data['acquisition_vals']
        best_x_to_probe = frame_data['best_x_to_probe']
        best_f_so_far = frame_data['best_f_so_far']
        
        # Clear previous traces
        self.fig.data = []
        
        # Add summary table at the top
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Iteration', 'Best Value Found', 'Best X Found', 'Successful Evaluations', 'Timed Out Evaluations'],
                        [iteration, f'{best_f_so_far:.2f}', f'{best_x_to_probe:.2f}', 
                         len(successful_evals), len(timed_out_evals)]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=1, col=1
        )
        
        # Extract data
        x_success = np.array([s[0] for s in successful_evals])
        y_success = np.array([s[1] for s in successful_evals])
        x_timed_out = np.array(timed_out_evals)
        
        # Plot 1: Objective Function
        self.fig.add_trace(
            go.Scatter(
                x=self.x_domain,
                y=[true_objective_function(x) for x in self.x_domain],
                name='True f(x) (Hidden)',
                line=dict(dash='dash', color='gray'),
                hovertemplate='x: %{x:.2f}<br>True f(x): %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        if gp_regressor is not None and len(gp_regressor.X_train_) > 0:
            y_pred, sigma = gp_regressor.predict(self.x_domain.reshape(-1, 1), return_std=True)
            self.fig.add_trace(
                go.Scatter(
                    x=self.x_domain,
                    y=y_pred,
                    name='GP Mean Prediction',
                    line=dict(color='blue'),
                    hovertemplate='x: %{x:.2f}<br>Predicted f(x): %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            self.fig.add_trace(
                go.Scatter(
                    x=np.concatenate([self.x_domain, self.x_domain[::-1]]),
                    y=np.concatenate([y_pred + sigma, (y_pred - sigma)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='GP Uncertainty',
                    hovertemplate='x: %{x:.2f}<br>Uncertainty: ±%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add successful and timed out points
        self.fig.add_trace(
            go.Scatter(
                x=x_success,
                y=y_success,
                mode='markers',
                name='Successful Eval',
                marker=dict(color='blue', size=10, line=dict(color='black', width=1)),
                hovertemplate='x: %{x:.2f}<br>f(x): %{y:.2f}<br>Status: Successful<extra></extra>'
            ),
            row=2, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=x_timed_out,
                y=np.full_like(x_timed_out, -900),
                mode='markers',
                name='Timed Out Eval',
                marker=dict(symbol='x', color='red', size=12, line=dict(width=2)),
                hovertemplate='x: %{x:.2f}<br>Status: Timed Out<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add best value line
        self.fig.add_trace(
            go.Scatter(
                x=[self.x_domain[0], self.x_domain[-1]],
                y=[best_f_so_far, best_f_so_far],
                name=f'Best f(x) so far: {best_f_so_far:.2f}',
                line=dict(color='green', dash='dot'),
                hovertemplate='Best value found: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 2: Halting Probability
        self.fig.add_trace(
            go.Scatter(
                x=self.x_domain,
                y=[halting_probability_function(x) for x in self.x_domain],
                name='True P(halts|x) (Hidden)',
                line=dict(dash='dash', color='gray'),
                hovertemplate='x: %{x:.2f}<br>True P(halts): %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        if gp_classifier is not None and hasattr(gp_classifier, 'classes_') and len(gp_classifier.classes_) > 1:
            x_domain_scaled = gp_classifier.scaler_.transform(self.x_domain.reshape(-1, 1))
            prob_pred = gp_classifier.predict_proba(x_domain_scaled)[:, 1]
            self.fig.add_trace(
                go.Scatter(
                    x=self.x_domain,
                    y=prob_pred,
                    name='GP Classifier P(halts)',
                    line=dict(color='red'),
                    hovertemplate='x: %{x:.2f}<br>Predicted P(halts): %{y:.2f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Add halted and timed out points
        self.fig.add_trace(
            go.Scatter(
                x=x_success,
                y=np.ones_like(x_success),
                mode='markers',
                name='Halted Eval (Y=1)',
                marker=dict(symbol='circle', color='blue', size=8, line=dict(color='black', width=1)),
                hovertemplate='x: %{x:.2f}<br>Status: Halted<extra></extra>'
            ),
            row=3, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=x_timed_out,
                y=np.zeros_like(x_timed_out),
                mode='markers',
                name='Timed Out Eval (Y=0)',
                marker=dict(symbol='x', color='red', size=12, line=dict(width=2)),
                hovertemplate='x: %{x:.2f}<br>Status: Timed Out<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Plot 3: Acquisition Function
        self.fig.add_trace(
            go.Scatter(
                x=self.x_domain,
                y=acquisition_vals,
                name='Risk-Aware Acquisition Function',
                line=dict(color='purple'),
                hovertemplate='x: %{x:.2f}<br>Acquisition Value: %{y:.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Add next probe point
        self.fig.add_trace(
            go.Scatter(
                x=[best_x_to_probe],
                y=[np.max(acquisition_vals)],
                mode='markers',
                name=f'Next Probe x: {best_x_to_probe:.2f}',
                marker=dict(symbol='star', color='green', size=15),
                hovertemplate='Next probe point:<br>x: %{x:.2f}<br>Acquisition Value: %{y:.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Update layout with explanations
        self.fig.update_layout(
            title=dict(
                text='Interactive Optimization Visualization',
                y=0.99,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    text="The AI is learning to optimize while avoiding computational timeouts.<br>" +
                         "It balances finding high values (top plot) with avoiding timeouts (middle plot)<br>" +
                         "to make smart decisions about where to probe next (bottom plot).",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.01,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="lightgray",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        # Update y-axis labels
        self.fig.update_yaxes(title_text="Objective Value", row=2, col=1)
        self.fig.update_yaxes(title_text="P(halts|x)", row=3, col=1)
        self.fig.update_yaxes(title_text="Acquisition Value", row=4, col=1)
        self.fig.update_xaxes(title_text="x", row=4, col=1)
        
        return self.fig

def create_interactive_visualization(optimization_history: List[Tuple], x_domain: np.ndarray,
                                   save_path: Optional[str] = None):
    """
    Create an interactive visualization of the optimization process.
    
    Args:
        optimization_history: List of tuples containing optimization state at each iteration
        x_domain: Domain points for plotting
        save_path: Path to save the interactive HTML file
    """
    visualizer = InteractiveOptimizationVisualizer(x_domain)
    fig = visualizer.create_interactive_plot(optimization_history[-1])  # Show final state
    
    if save_path:
        pio.write_html(fig, save_path)
    
    return fig 