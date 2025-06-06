## The Uncomputable Oracle Simulator (Phase 1)

### `true_objective_function(x)`: This is the ideal information. If we lived in a world where every computation instantly returned a value, this is what we'd optimize directly. By making it simple (a quadratic), we have a known optimum to compare against. Crucially, the AI optimizer will not have direct access to this function. 

### `halting_probability_function(x)`: This is our simulation of the "uncomputability" or "computational difficulty" of different inputs. 
- We used `np.sin` and `np.cos` to create some continuous variation in probability
- We introduced some "danger" (low `P(halts|x)`) and "safe" zones (high `P(halts|x)`)
- `np.clip` ensures the probabilities remain between 0 and 1
Just like the above, the AI does not know this function. It must learn an approximation of this behavior from the timeouts it experiences.

### `uncomputable_oracle(x,...)`: This is the only way our AI can get information about the objective function. It mimics a real-world black-box system. There are finite computational budget and the real-world cost of an evaluation. Without this, the problem would just be standard optimization. 
- `random.random() < halt_chance` introduces the probabilistic nature. Even if `P(halts|x)` is high, there's still a high chance of timing out, and vice-versa. This is critical for uncertainty management.
- There's a penalty value for timeouts, and it directly translates the "uncomputable" nature into a measurable (and undesirable) outcome for the optimizer.

### `collect_initial_data()`: Our AI can't start optimizing if it has no data. This function provides "initial experiences" it gathers from the world. For initial exploration, random sampling is a common and simple strategy -> ensures some initial diversity across the domain.

## Building the AI's Brain: Surrogate Models (Phase 2)
In this phase, the AI will attempt to learn two distinct but related things from the data it collects:
- **What is the likely value of $f(x)$ for inputs that do halt?**
- **What is the probability that a given input $x$ will halt?**

These two models are the foundation of our AI's understanding of the problem space. They allow the AI to move beyond random guessing and make informed decisions about where to probe next.

### **Gaussian Processes (GPs), The "Probabilistic Brain"**: A powerful choice because they don't just give single prediction (like a linear regression model), instead, they provide a probability distribution over possible  function values at any given point. This means for each $x$, we get a mean prediction (the most likely value) and a standard deviation (uncertainty). The uncertainty is crucial for our AI to decide  whether to explore or exploit.

### **Kernels**: The `kernel` parameter defines the expected smoothness or shape of the functions we're trying to learn.
- `Matern` kernel: versatile choice; often robust for functions that aren't smooth
- `WhiteKernel`: Account for noise in the observations (measurement noise)
- `ConstantKernel`: scales the output of the other kernels

### **Gaussian Process Regressor**: Learning $f(x)$, this model exclusively learns from successful evaluations. If an evaluations times out, we don't know what $f(x)$  would have been, so it cannot inform the model directly.

### **Gaussian Process Classifier**: Learning $P(halts|x)$, this model tackles the "uncomputability" directly. It learns to classify inputs from two categories: "halted" (labeled as 1), and "timed out" (labeled as 0). From this classification, it can provide the probability of halting.

Now, our AI has the ability to interpret the results of its probes and build internal probabilistic models of the world. This is a huge leap from simply randomly trying inputs. 

## The Decision Maker - Acquisition Function (Phase 3)
This is where our AI truly becomes "smart". Having built its understanding of the world using the surrogate models (GP Regressor and GP Classifier), the AI now needs a strategy to pick the next best point to evaluate. This strategy is embodied in the acquisition function. It quantifies the "value" of trying out an untested point, balancing finding high objective values with exploring uncertain regions and, crucially, avoiding areas likely to time out.

### Expected Improvement (EI): A standard in Bayesian Optimization, and then adapt it to incorporate the halting probability predicted by our GP Classifier. The core idea of is to select points that are expected to yield a better objective value than the best one found so far, taking into account the uncertainty of the surrogate model. Our twist is to multiply this by the probability that the evaluation will even halt in the first place.