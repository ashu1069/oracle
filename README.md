# Uncomputable Oracle Optimizer

A Bayesian optimization framework for handling computationally expensive and potentially uncomputable objective functions. This project demonstrates how to build an intelligent optimizer that can handle the uncertainty of computation itself.

## Motivation: Optimization Meets Uncomputability

**The Problem:**  
Imagine an optimization problem where the objective function is *partially uncomputable* or *extremely expensive* to compute for certain inputs. For example, the value of the objective function might depend on the halting status of a Turing machine, or require an extremely long simulation. How would an AI approach such a problem? This project delves into the intersection of **computability theory** and **optimization**.

Traditional optimization assumes that every evaluation of $f(x)$ returns a value. But what if, for some $x$, the computation never finishes, or takes so long that it's practically impossible to obtain the result? This is not just a theoretical curiosity—many real-world problems (e.g., complex simulations, undecidable computations, or black-box scientific models) exhibit this property.

**Uncomputable Oracle Optimizer** is a framework and educational tool that explores how an intelligent optimizer can:
- Model and learn which regions of the input space are "safe" (computable) and which are "dangerous" (uncomputable or prohibitively expensive)
- Balance risk and reward by making decisions that maximize expected improvement while avoiding computational dead-ends
- Leverage probabilistic models (Gaussian Processes) to reason about both the objective function and the probability of successful computation

## The Origin Story

This project was born from a fascinating challenge in real-world optimization: what happens when computation itself is uncertain? Traditional optimization assumes that evaluating f(x) always returns a value, but in reality, some computations might:
- Take too long to complete
- Never finish at all
- Have varying computation times

The Uncomputable Oracle Optimizer tackles this challenge by treating computation as an oracle - a mysterious black box that might or might not return a value. This approach combines concepts from:
- Computability Theory (the study of what can and cannot be computed)
- Bayesian Optimization (probabilistic modeling of unknown functions)
- Risk-Aware Decision Making (balancing potential rewards against computational risks)

## The Challenge: Exploring the Unknown

Imagine you're a data scientist exploring a mysterious dataset. Each data point (x) can potentially yield a valuable insight (f(x)). However, there's a catch: probing some data points causes your analysis software to "freeze" indefinitely. You have a limited computational budget, and your goal is to find the data point with the highest f(x) value that can be successfully analyzed, while intelligently avoiding the "freezing" points.

This is the essence of the Uncomputable Optimizer: an AI that learns to navigate the delicate balance between:
- Seeking valuable insights (high f(x) values)
- Avoiding computational dead-ends (points that never complete)
- Managing limited computational resources

## Key Features

- Risk-Aware Bayesian Optimization: Combines traditional Bayesian optimization with computational risk assessment
- Gaussian Process Models: Uses probabilistic models to learn both the objective function and computation success probability
- Visualization Tools: Rich visualization of the optimization process and model beliefs
- Educational Focus: Designed to be both practical and educational, with detailed documentation and examples

## Tutorial: Building an Intelligent Optimizer

This project serves as a comprehensive tutorial on building an optimizer from a statistical machine learning perspective. It covers:

1. **Phase 1**: Simulating computational uncertainty
   - Creating an oracle that may or may not return values
   - Modeling halting probability
   - Handling timeouts and penalties

2. **Phase 2**: Building probabilistic surrogate models
   - Gaussian Process Regression for objective function
   - Gaussian Process Classification for halting probability
   - Uncertainty quantification

3. **Phase 3**: Making risk-aware decisions
   - Expected Improvement with halting probability
   - Balancing exploration and exploitation
   - Risk-aware acquisition functions

4. **Phase 4**: Putting it all together
   - The optimization loop
   - Visualization and analysis
   - Real-world applications

## Quick Start

```bash
pip install uncomputable-optimizer
```

Basic usage:

```python
from uncomputable_optimizer import UncomputableOptimizer

# Define your objective function and computation time limit
optimizer = UncomputableOptimizer(
    objective_function=your_function,
    time_limit=0.1,  # seconds
    penalty_value=-1000.0
)

# Run optimization
best_x, best_f = optimizer.optimize(
    num_initial_samples=5,
    num_iterations=20,
    x_min=0.0,
    x_max=10.0
)
```

## Documentation

For detailed documentation, including tutorials and API reference, visit our [documentation site](https://uncomputable-optimizer.readthedocs.io/).

## Learning Resources

- [Tutorial Notebooks](notebooks/): Step-by-step guides on using the optimizer
- [Conceptual Guide](docs/concepts.md): Deep dive into the mathematical and conceptual foundations
- [API Reference](docs/api.md): Detailed API documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by the challenges of real-world optimization problems where computation itself is uncertain. Special thanks to the Bayesian optimization and Gaussian process communities for their foundational work.

## Contact

For questions and feedback, please [open an issue](https://github.com/yourusername/uncomputable-optimizer/issues) or contact the maintainers.

## Technical Details

### The Uncomputable Oracle Simulator (Phase 1)

#### `true_objective_function(x)`
This is the ideal information. If we lived in a world where every computation instantly returned a value, this is what we'd optimize directly. By making it simple (a quadratic), we have a known optimum to compare against. Crucially, the AI optimizer will not have direct access to this function. 

#### `halting_probability_function(x)`
This is our simulation of the "uncomputability" or "computational difficulty" of different inputs. 
- We used `np.sin` and `np.cos` to create some continuous variation in probability
- We introduced some "danger" (low `P(halts|x)`) and "safe" zones (high `P(halts|x)`)
- `np.clip` ensures the probabilities remain between 0 and 1

Just like the above, the AI does not know this function. It must learn an approximation of this behavior from the timeouts it experiences.

#### `uncomputable_oracle(x,...)`
This is the only way our AI can get information about the objective function. It mimics a real-world black-box system. There are finite computational budget and the real-world cost of an evaluation. Without this, the problem would just be standard optimization. 
- `random.random() < halt_chance` introduces the probabilistic nature. Even if `P(halts|x)` is high, there's still a high chance of timing out, and vice-versa. This is critical for uncertainty management.
- There's a penalty value for timeouts, and it directly translates the "uncomputable" nature into a measurable (and undesirable) outcome for the optimizer.

#### `collect_initial_data()`
Our AI can't start optimizing if it has no data. This function provides "initial experiences" it gathers from the world. For initial exploration, random sampling is a common and simple strategy -> ensures some initial diversity across the domain.

### Building the AI's Brain: Surrogate Models (Phase 2)

In this phase, the AI will attempt to learn two distinct but related things from the data it collects:
- What is the likely value of $f(x)$ for inputs that do halt?
- What is the probability that a given input $x$ will halt?

These two models are the foundation of our AI's understanding of the problem space. They allow the AI to move beyond random guessing and make informed decisions about where to probe next.

#### Gaussian Processes (GPs), The "Probabilistic Brain"
A powerful choice because they don't just give single prediction (like a linear regression model), instead, they provide a probability distribution over possible function values at any given point. This means for each $x$, we get a mean prediction (the most likely value) and a standard deviation (uncertainty). The uncertainty is crucial for our AI to decide whether to explore or exploit.

#### Kernels 
The `kernel` parameter defines the expected smoothness or shape of the functions we're trying to learn.
- `Matern` kernel: versatile choice; often robust for functions that aren't smooth
- `WhiteKernel`: Account for noise in the observations (measurement noise)
- `ConstantKernel`: scales the output of the other kernels

#### Gaussian Process Regressor 
Learning $f(x)$, this model exclusively learns from successful evaluations. If an evaluations times out, we don't know what $f(x)$ would have been, so it cannot inform the model directly.

#### Gaussian Process Classifier 
Learning $P(halts|x)$, this model tackles the "uncomputability" directly. It learns to classify inputs from two categories: "halted" (labeled as 1), and "timed out" (labeled as 0). From this classification, it can provide the probability of halting.

### The Decision Maker - Acquisition Function (Phase 3)

This is where our AI truly becomes "smart". Having built its understanding of the world using the surrogate models (GP Regressor and GP Classifier), the AI now needs a strategy to pick the next best point to evaluate. This strategy is embodied in the acquisition function. It quantifies the "value" of trying out an untested point, balancing finding high objective values with exploring uncertain regions and, crucially, avoiding areas likely to time out.

#### Expected Improvement (EI)
A standard in Bayesian Optimization, and then adapt it to incorporate the halting probability predicted by our GP Classifier. The core idea of is to select points that are expected to yield a better objective value than the best one found so far, taking into account the uncertainty of the surrogate model. Our twist is to multiply this by the probability that the evaluation will even halt in the first place.

#### The Role of Acquisition Function
This is the AI's "strategy" or "policy". Given its current understanding (the models), what's the optimal next action? It balances different desires:
- **Exploitation**: Probing points predicted to have very high values
- **Exploration**: Probing points where the model is very uncertain (high standard deviation in GP regressor), as these might reveal surprising high values or new "safe" zones.
- **Risk Aversion**: Avoiding points likely to time out (low `P(halts|x)`), even if they appear promising otherwise.

#### Calculation of Expected Improvement (EI)
- EI quantifies how much better we expect a new sample to be compared to the `f_best` value found so far.
- Z-score normalizes the difference between the predicted value and the best-found value by the uncertainty. A high Z-score means a point is either predicted to be much better or there's high uncertainty that could lead to a much better value.
- `norm.cdf(Z)` gives the probability that a random variable from a standard normal distribution is less than or equal to Z; and `norm.pdf(Z)` gives the height of the standard normal distribution at Z.

#### Risk-Aware Acquisition
We combine the traditional EI with the probability of the evaluation actually succeeding. By multiplying `P(halts|x)` by `EI(x)`, we effectively penalize points that are likely to time out. A point might have a very high `EI`, but if its `P(halts|x)` is near zero, its risk-aware acquisition will also be near zero, discouraging the AI from picking it.

By maximizing this risk-aware acquisition function, our AI will make decisions that optimally balance the desire for high objective values, the need to explore uncertain regions, and the imperative to avoid time-consuming failures. This is a crucial step towards our "optimizer as a product."

### The Optimization Loop (Phase 4)