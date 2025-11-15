# Optimizers: SGD with Momentum, RMSprop, and Adam

## What is an Optimizer?

An **optimizer** is an algorithm that updates a neural network's weights to minimize the loss function. It decides how much to change each weight based on the gradients (slopes) from backpropagation.

### The Problem with Basic Gradient Descent

- If the slope is steep, you take a big step
- If the slope is gentle, you take a small step
- You always go downhill

**Problems:**
1. **Slow in flat regions**: Tiny gradients = tiny steps = slow progress
2. **Oscillations in steep regions**: Alternates between overshooting left and right
3. **Same learning rate everywhere**: Can't adapt to different parts of the landscape

Modern optimizers solve these problems well.

---

## Minibatch Stochastic Gradient Descent (SGD)

Instead of computing gradients on the entire dataset (slow) or one example at a time (noisy), we use **minibatches** - small random subsets of data.

### Why Minibatches?

**Full Batch:**
- Gradient: Very accurate
- Speed: Very slow (must process entire dataset)
- Memory: May not fit in RAM

**Single Example (SGD):**
- Gradient: Very noisy
- Speed: Fast per step, but needs many steps
- Memory: Efficient

**Minibatch (Best of Both):**
- Gradient: Reasonably accurate
- Speed: Fast (parallel GPU computation)
- Memory: Manageable
- Typical batch sizes: 32, 64, 128, 256

### How It Works

Each epoch:
1. Shuffle the training data
2. Split into batches (e.g., 128 examples each)
3. For each batch:
   - Forward pass → compute loss
   - Backward pass → compute gradients
   - Update weights
4. Repeat

---

## SGD with Momentum

**Problem:** Basic SGD oscillates in narrow valleys and moves slowly in flat regions.

**Solution:** Add "momentum" - remember the direction you've been moving and keep going that way.

### Intuition

Imagine rolling a ball down a hill:
- The ball doesn't just follow the slope instantly
- It builds up speed (momentum) in consistent directions
- It dampens oscillations because momentum from opposite directions cancel out
- It powers through small bumps and flat regions

### The Math

```
velocity = β * velocity + gradient
parameter = parameter - learning_rate * velocity
```

Where:
- **β (beta)**: Momentum coefficient (typically 0.9)
  - β = 0: No momentum (regular SGD)
  - β = 0.9: Keep 90% of previous direction (common)
  - β = 0.99: Very smooth, keeps 99% of history

### What It Does

1. **Accelerates in consistent directions**: If gradients keep pointing the same way, velocity builds up
2. **Dampens oscillations**: Opposing gradients cancel out in the velocity
3. **Crosses flat regions faster**: Accumulated velocity carries you through

### When to Use

- Good for: Problems with narrow valleys or saddle points
- Learning rate: Can use higher LR than vanilla SGD (momentum smooths out updates)
- Typical β: 0.9

---

## RMSprop (Root Mean Square Propagation)

**Problem:** Different parameters need different learning rates. Some features appear rarely (large updates needed), others appear frequently (small updates needed).

**Solution:** Adapt the learning rate for each parameter based on recent gradient history.

### Intuition

Imagine training on images:
- **Common feature** (e.g., edge detection): Gradients are consistent and large → divide by large number → smaller effective learning rate
- **Rare feature** (e.g., specific texture): Gradients are small and infrequent → divide by small number → larger effective learning rate

This prevents the optimizer from "forgetting" rare but important features.

### The Math

```
cache = β * cache + (1 - β) * gradient²
parameter = parameter - learning_rate * gradient / (√cache + ε)
```

Where:
- **cache**: Moving average of squared gradients
- **β (beta)**: Decay rate (typically 0.9)
- **ε (epsilon)**: Small constant (1e-8) to prevent division by zero

### What It Does

1. **Normalizes updates**: Parameters with large gradients get smaller updates, small gradients get larger updates
2. **Per-parameter learning rates**: Each weight has its own effective learning rate
3. **Handles sparse gradients well**: Rare features don't get overwhelmed by common ones

### When to Use

- Good for: Problems with sparse features or varying gradient scales
- Learning rate: Can use smaller base LR (e.g., 0.001) because it's adaptive
- Typical β: 0.9

---

## Adam (Adaptive Moment Estimation)

**Problem:** Can we get the best of both worlds - momentum and adaptive learning rates?

**Solution:** Adam combines momentum (first moment) and RMSprop (second moment).

### Intuition

Adam is like a smart hiker:
1. **Momentum part**: Remembers which direction you've been walking (smooth path)
2. **RMSprop part**: Adjusts step size based on terrain (small steps on steep cliffs, big steps on gentle slopes)
3. **Bias correction**: Accounts for the fact that at the start, you haven't built up enough history

### The Math

```
# First moment (momentum) - moving average of gradients
m = β₁ * m + (1 - β₁) * gradient

# Second moment (RMSprop) - moving average of squared gradients
v = β₂ * v + (1 - β₂) * gradient²

# Bias correction (important in early iterations)
m_hat = m / (1 - β₁^t)
v_hat = v / (1 - β₂^t)

# Update
parameter = parameter - learning_rate * m_hat / (√v_hat + ε)
```

Where:
- **m**: First moment estimate (like momentum)
- **v**: Second moment estimate (like RMSprop)
- **β₁**: Decay for first moment (typically 0.9)
- **β₂**: Decay for second moment (typically 0.999)
- **t**: Timestep counter
- **ε**: Small constant (1e-8)

### Bias Correction - Why?

In the first few iterations:
- m and v are initialized to zero
- They're biased toward zero (haven't accumulated history yet)
- Dividing by (1 - β^t) corrects this
- As t → ∞, the correction term → 1 (becomes negligible)

### What It Does

1. **Smooth updates**: Momentum prevents jittery movements
2. **Adaptive learning rates**: Different LR for each parameter
3. **Fast convergence**: Combines benefits of both techniques
4. **Works out-of-the-box**: Default hyperparameters work well for many problems

### When to Use

- Good for: Almost everything! It's the most popular optimizer
- Learning rate: Typically 0.001 (less sensitive to LR choice)
- Typical β₁: 0.9
- Typical β₂: 0.999

---

## Comparison Summary

| Optimizer | Strengths | Weaknesses | When to Use |
|-----------|-----------|------------|-------------|
| **SGD Momentum** | Simple, interpretable, good for convex problems | Requires careful LR tuning, struggles with adaptive needs | Small networks, well-understood problems |
| **RMSprop** | Handles sparse gradients well, per-parameter LR | No momentum, can be unstable on non-stationary problems | RNNs, time-series, sparse data |
| **Adam** | Combines momentum + adaptive LR, robust, works well out-of-the-box | More hyperparameters, slightly more computation | Default choice for most problems |

---


### Learning Rate

- **SGD Momentum**: Start with 0.01, tune carefully
- **RMSprop**: Start with 0.001
- **Adam**: Start with 0.001 (usually works)

### Batch Size

- Smaller batches (32-64): More noise, better generalization, slower
- Larger batches (128-256): Less noise, faster training, may generalize worse
- Try 128 as a default

### Convergence

- **Fast initial drop**: Good sign, optimizer is working
- **Oscillating loss**: Learning rate too high or batch size too small
- **Plateaus early**: Learning rate too low or stuck in local minimum
- **Adam converges fastest**: Usually reaches high accuracy in fewer epochs


## Summary

1. **Minibatch SGD**: Balances computation speed and gradient accuracy
2. **Momentum**: Smooths updates and accelerates in consistent directions
3. **RMSprop**: Adapts learning rate per parameter based on gradient history
4. **Adam**: Combines momentum and adaptive LR for robust performance
5. **Default choice**: Adam with LR=0.001 works well for most problems
6. **When in doubt**: Try all three and compare convergence curves!
