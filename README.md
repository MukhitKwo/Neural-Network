# Neural Network Simulation

A 2D neuroevolution simulation built from scratch using Python, NumPy, and Pygame. Agents are controlled by a multilayer perceptron (MLP) and evolve over generations using a genetic algorithm to find and collect goals as efficiently as possible.

![demo](https://github.com/user-attachments/assets/b2a442f3-de5a-4522-9855-01943cff78f1)

---

## How It Works

Each generation, a population of agents navigates a 2D environment and tries to collect all goals. Agents are controlled entirely by a neural network that takes the agent's current state as input and outputs acceleration values. There are no hardcoded rules or pathfinding, all behavior emerges purely through evolution. Agents start blue and shift toward red as they collect more goals, giving a quick visual read of how well each agent is performing.

At the end of each generation, agents are ranked by fitness. The best agents are carried over unchanged (elitism), and the rest of the population is bred through tournament selection. Offspring inherit the weights of their selected parent, with small random mutations applied to introduce variation. This cycle repeats generation after generation until the population converges into an effective strategy.

Goals are randomized at the start of each generation, so agents must generalize rather than memorize positions.

---

## Controls

| Key | Action |
|---|---|
| `Space` | Pause / unpause |
| `↑` / `↓` | Increase / decrease simulation speed |
| `L` | Toggle proximity lines (agent to closest goal) |
| `F` | Toggle agent fitness value |
| `→` | Step forward one frame (while paused) |
| `Esc` | Quit |

---

## Configuration

All simulation parameters are defined in `config.toml`:

```toml
[simulation]
window_width          # Width of the simulation window in pixels
window_height         # Height of the simulation window in pixels
steps_per_generation  # How many steps each generation runs for

[population]
agent_population_size # Number of agents per generation
goal_population_size  # Number of goals per generation

[agent]
max_velocity          # Maximum speed per axis
max_acceleration      # Maximum acceleration output from the network

[reproduction]
top_agents            # Number of elite agents carried over unchanged
sample_size           # Tournament size for selection

[fitness]
proximity_multiplier  # Scales the per-step proximity reward
time_multiplier       # Scales the time bonus on goal collection
goals_reached_bonus   # Flat reward for collecting a goal

[network]
hidden_layer_dimensions  # List defining the size of each hidden layer
mutation_rate            # Standard deviation of the Gaussian mutation noise
mutation_probability     # Per-weight probability of mutation being applied
```

Performance note: The simulation is not optimized for large populations. If your machine starts to struggle, the first thing to lower is agent_population_size in config.toml.

---

## How to Run

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

**Install dependencies:**
```bash
uv sync
```

**Run the simulation:**
```bash
uv run simulation.py
```

---

## Math

### MLP Forward Pass

A neural network is composed of a series of layers, each containing a number of neurons.

Each neuron has one weight per input it receives, plus a bias term. For example, if a layer receives 5 inputs, each neuron in that layer has 5 weights and 1 bias. Both weights and biases are initialized with random values between -1 and 1, but can drift beyond that range as evolution progresses.

The network receives normalized inputs where possible.

For each neuron ($n$) in a layer, all inputs ($x$) are multiplied by their respective weights ($w$), summed together, and added to a bias ($b$). This gives the pre-activation value ($Z$):

$$Z_{n} = (x_{1} \cdot w_{n1}) + (x_{2} \cdot w_{n2}) + ... + (x_i \cdot w_{ni}) + b_{n}$$

or more compactly using the dot product:

$$Z_{n} = \mathbf{w}_n \cdot \mathbf{x} + b_n$$

After computing the pre-activation value, an activation function is applied to produce the neuron's output ($A$), keeping it bounded between -1 and 1. This network uses tanh:

$$ A_n = \tanh(Z_{n}) = \frac{e^{Z_n} - e^{-Z_n}}{e^{Z_n} + e^{-Z_n}} $$

The activations of one layer become the inputs of the next, and this process repeats for every layer in the network.

The network takes 5 inputs and produces 2 outputs ($a_x$, $a_y$):

| Input | Description |
|---|---|
| $\frac{1}{1 + d}$ | Closeness to the nearest goal (where $d = \sqrt{dx^2 + dy^2}$) |
| $dx / d$ | Normalized x direction to the nearest goal |
| $dy / d$ | Normalized y direction to the nearest goal |
| $v_x / M_\text{velocity}$ | Current x velocity, normalized |
| $v_y / M_\text{velocity}$ | Current y velocity, normalized |

Where $dx$ and $dy$ are the x and y distances to the nearest goal, each divided by the window dimensions.

The final output is the last layer's activation scaled by the maximum allowed acceleration:

$$\text{Output}_n = A_n \times M_{\text{acceleration}}$$

The output layer is appended automatically, so `hidden_layer_dimensions` in `config.toml` only needs to define the intermediate layers.


### Fitness Function

Fitness is accumulated throughout the generation from two sources: a bonus for collecting goals, and a continuous reward for staying close to the nearest goal.

$$Fitness = \underbrace{ \left(\sum_{g=0}^{G_\text{reached}} T_\text{multiplier} \cdot \frac{S_\text{max} - s_g}{S_\text{max}} + R_\text{bonus} \right)}_{\text{goal collection bonus}} + \underbrace{ \left( P_\text{multiplier} \sum_{s=1}^{S_\text{max}} \frac{1}{\sqrt{\Delta x^2 + \Delta y^2}} \right) }_{\text{proximity bonus}}$$ 

Where:
- $G_{\text{reached}}$ is the number of goals collected during the generation
- $s_g$ is the step at which a given goal was collected (earlier is better)
- $T_{\text{multiplier}}$ scales the time bonus for collecting goals early
- $R_{\text{bonus}}$ is a flat reward added for each goal collected
- $P_{\text{multiplier}}$ scales down the proximity sum so it does not dominate the total
- $S_{\text{max}}$ is the maximum number of steps per generation

### Mutation

Each weight and bias has an independent probability ($p$) of being mutated each generation. When selected, a value sampled from a Gaussian distribution with standard deviation $\sigma$ is added to it:

$$w \leftarrow w + \mathcal{N}(0,\ \sigma) \quad \text{with probability } p$$

Where $\sigma$ is the mutation rate, which controls how large the applied noise can be.

### Reproduction

At the end of a generation, the best agents are selected to produce the next population. The top-performing agents (by fitness) are carried over directly without any mutation, preserving the best solutions found so far.

For the rest of the new population, each slot is filled using tournament selection: a random sample of $K$ agents is drawn, the best of that sample is chosen as the parent, and its weights are copied and mutated to create the offspring.
