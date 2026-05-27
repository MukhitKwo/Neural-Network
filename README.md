# Neural Network Simulation

A 2D neuroevolution simulation built from scratch using Python, NumPy, and Pygame. Agents are controlled by a multilayer perceptron (MLP) and evolve over generations using a genetic algorithm to find and collect goals as efficiently as possible.

<!-- Add gif here -->

---

## How It Works

Each generation, a population of agents navigates a 2D environment and tries to collect all goals. Agents are controlled entirely by a neural network that takes the agent's current state as input and outputs acceleration values. No hardcoded rules or pathfinding, behavior emerges purely through evolution.

At the end of each generation, agents are ranked by fitness. The best agents are carried over unchanged (elitism), and the rest of the population is bred through tournament selection. Offspring inherit the weights of their selected parent, with a small random mutations applied to introduce variation. This cycle repeats until the population converges on an effective strategy.

Goals are randomized at the start of each generation, so agents must generalize rather than memorize positions.

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
proximity_multiplier      # Scales the per-step proximity reward
time_bonus_multiplier     # Scales the time bonus on goal collection
goals_reached_multiplier  # Flat reward for collecting a goal

[network]
hidden_layer_dimensions  # List defining the size of each hidden layer, e.g. [10, 8]
mutation_rate            # Standard deviation of the Gaussian mutation noise
mutation_probability     # Per-weight probability of mutation being applied
```

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

Each layer computes a weighted sum of its inputs, adds a bias, and passes the result through a tanh activation function:

$$a^{(l)} = \tanh(W^{(l)} \cdot a^{(l-1)} + b^{(l)})$$

Where:
- $W^{(l)}$ is the weight matrix for layer $l$
- $b^{(l)}$ is the bias vector for layer $l$
- $a^{(l-1)}$ is the activation from the previous layer (or the input vector for $l=1$)

The network takes 5 inputs and produces 2 outputs ($a_x$, $a_y$):

| Input | Description |
|---|---|
| $dx / \|d\|$ | Normalized x direction to closest goal |
| $dy / \|d\|$ | Normalized y direction to closest goal |
| $1 / (1 + \|d\|)$ | Proximity to closest goal |
| $v_x$ | Current x velocity |
| $v_y$ | Current y velocity |

### Fitness Function

Fitness is accumulated over the course of a generation:

$$F = \sum_{\text{goals collected}} \left( R_{\text{goal}} + \frac{T - t}{T} \cdot R_{\text{time}} \right) + \sum_{\text{steps}} \min\left( \frac{P}{d + \epsilon},\ P_{\text{max}} \right)$$

Where:
- $R_{\text{goal}}$ is the flat reward for collecting a goal
- $T$ is the total steps per generation, $t$ is the step at which the goal was collected
- $R_{\text{time}}$ is the time bonus multiplier (rewards collecting goals early)
- $P$ is the proximity multiplier, $d$ is the distance to the closest goal
- $P_{\text{max}}$ is the per-step proximity cap (prevents orbiters from accumulating fitness without collecting)

### Mutation

Each weight and bias has an independent probability $p_m$ of being mutated. When selected, a value sampled from a Gaussian distribution is added:

$$w \leftarrow w + \mathcal{N}(0,\ \sigma^2) \quad \text{with probability } p_m$$

Where $\sigma$ is the mutation rate (standard deviation of the noise).