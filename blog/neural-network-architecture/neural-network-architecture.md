# Neural network architecture: layers, shapes, and learning

Chanupa Deshan · September 6, 2026 · Foundations + NumPy example

A neural network architecture is the blueprint that says how information moves through a model: what enters, which layers transform it, how wide and deep those layers are, and what comes out. Training does not invent this blueprint. Training adjusts the weights inside it.

This guide builds a small multilayer perceptron from first principles, follows one tensor through every layer, counts its parameters, and connects the architecture to the loss and learning process.

## 01 · The basic building block

A neuron receives inputs, multiplies them by learned weights, adds a learned bias, and applies an activation function. For an input vector `x`, one neuron computes `z = x · w + b` and then `a = f(z)`.

The weights control how strongly each input contributes. The bias shifts the neuron’s threshold. The activation makes the network nonlinear. Without nonlinear activations between layers, stacking many dense layers still collapses into one linear transformation and cannot learn nonlinear decision boundaries.

Real implementations process a batch as matrix operations. A dense layer with input matrix `X`, weight matrix `W`, and bias vector `b` computes:

```python
Z = X @ W + b
A = relu(Z)
```

If `X` has shape `(batch, inputs)` and the layer has `units` neurons, `W` has shape `(inputs, units)`, `b` has shape `(1, units)`, and the output has shape `(batch, units)`. Writing down these shapes catches many architecture errors before training begins.

## 02 · Input, hidden, and output layers

The input layer represents one example. Its width comes from the data representation: two numeric features give width 2; a flattened 28 × 28 grayscale image gives width 784; a token sequence is usually represented as `(sequence length, embedding size)` rather than one flat vector.

Hidden layers build intermediate representations. Width is the number of units or channels in a layer. Depth is the number of learned transformations along the path from input to output. A model with many parameters is not automatically deep: one very wide layer can contain more parameters than several narrow layers.

The output layer must match the task:

| Task | Output units | Typical output/loss pairing |
| --- | ---: | --- |
| Numeric regression | 1 or one per target | Linear output + mean squared error |
| Binary classification | 1 | Sigmoid + binary cross-entropy |
| Single-label multiclass | Number of classes | Softmax + categorical cross-entropy |
| Multi-label classification | Number of labels | One sigmoid per label + binary cross-entropy |

Softmax classes compete and probabilities sum to one. Independent sigmoids allow several labels to be active. The target encoding, output shape, and loss must agree.

## 03 · Follow a 2 → 4 → 1 network

Consider four examples in one batch, each with two features. Our architecture has a hidden layer with four ReLU units and one sigmoid output:

```text
X (4 × 2)
  → Dense: W1 (2 × 4), b1 (1 × 4)
  → ReLU:  H (4 × 4)
  → Dense: W2 (4 × 1), b2 (1 × 1)
  → Sigmoid: prediction (4 × 1)
```

The first layer has `2 × 4 + 4 = 12` trainable parameters. The output layer has `4 × 1 + 1 = 5`. The whole network has 17. In general, a dense layer with `n_in` inputs and `n_out` units has `n_in × n_out + n_out` parameters. The batch size changes how many examples are processed together; it does not change the parameter count.

Download [mlp_from_scratch.py](examples/mlp_from_scratch.py) and [requirements.txt](requirements.txt), then run:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python mlp_from_scratch.py
```

The script learns XOR, a tiny problem that a single linear boundary cannot solve. It is an educational implementation without production features such as validation splits, minibatching, checkpointing, or automatic differentiation.

## 04 · Activations shape what the network can represent

ReLU computes `max(0, z)`. It is inexpensive and often a useful default for hidden layers. A ReLU unit can become inactive if its inputs remain negative; variants such as leaky ReLU keep a small negative-side slope.

Sigmoid maps a number to `(0, 1)` and fits a binary probability output. Tanh maps to `(-1, 1)`. Sigmoid and tanh can saturate at large magnitudes, producing very small gradients. Softmax turns a vector of logits into a distribution over mutually exclusive classes.

Keep logits numerically stable. Training libraries usually provide combined losses such as cross-entropy from logits, which are safer than manually applying softmax or sigmoid and then taking logarithms. The NumPy example clips the sigmoid input and adds a small epsilon because it implements the operations directly for clarity.

## 05 · How the architecture learns

A forward pass sends a batch through every layer and produces predictions. The loss measures how those predictions differ from the targets. Backpropagation applies the chain rule from the loss back through the graph to calculate a gradient for every trainable parameter. An optimizer then changes the parameters, usually in the direction that reduces loss.

The architecture determines the computation graph and the available paths for gradients. Initialization determines the starting scale of signals. The example uses a variance-scaled random initialization for ReLU weights and zero biases. Initializing every weight to the same value would make units learn the same features because their gradients would remain symmetric.

The learning rate controls update size. A very large value may make training unstable; a very small value may make progress impractically slow. Epoch count, batch size, optimizer, regularization, and learning-rate schedule are training choices around the architecture, not substitutes for a sensible output and loss.

## 06 · Common architecture families

A multilayer perceptron uses dense connections and works well for fixed-size feature vectors. Parameter count grows quickly when a high-dimensional input connects to every hidden unit.

A convolutional neural network shares small filters across spatial positions. This makes it well suited to images and grid-like signals: nearby structure matters, and a useful pattern may appear in different locations.

Recurrent networks process ordered steps while carrying state forward. LSTMs and GRUs add gates that help manage longer dependencies, though sequential computation can limit parallelism.

Transformers use attention so each token can combine information from other tokens. Embeddings represent tokens, attention mixes information across positions, and feed-forward blocks transform each position. Residual connections and normalization help train deep stacks. Attention cost can grow rapidly with sequence length, so the context budget is an architectural and operational choice.

Autoencoders learn to reconstruct inputs through a constrained representation. Graph neural networks pass messages along edges. None of these families is universally best; the structure should reflect the relationships in the data and the prediction task.

## 07 · Depth, width, and regularization

More width can represent more features at a layer. More depth composes transformations and can express hierarchical structure efficiently. Both raise capacity and compute cost. A larger training score with a worse validation score suggests overfitting, while poor performance on both may indicate underfitting, unsuitable features, optimization trouble, or noisy labels.

Regularization controls how the capacity is used. Weight decay discourages large weights. Dropout randomly removes activations during training and is disabled during ordinary inference. Early stopping selects a checkpoint using validation performance. Normalization layers can stabilize activation scales; their behavior and trainable parameters depend on the chosen layer and framework.

Skip or residual connections add an earlier representation to a later one when shapes are compatible. They give information and gradients shorter routes through deep networks. If dimensions differ, the architecture needs a projection or another explicit alignment.

## 08 · Design and debug with a shape table

Before training, write one representative input shape and trace every layer. Count parameters, identify the activation after each learned transform, and confirm that the final output and loss match the target. Then run one batch and verify that every tensor is finite and has the expected shape.

Use a simple baseline and change one architectural choice at a time. Track training and validation loss, evaluate metrics suited to the task, and inspect errors. If the model cannot overfit a tiny clean sample, investigate the data, target encoding, forward pass, loss, gradients, and update step before adding more layers.

Check the following architecture contract:

- The input shape and preprocessing match what the first learned layer expects.
- Every matrix multiplication and residual addition has compatible dimensions.
- Hidden activations provide the nonlinearity the task needs.
- Output units, activation, target encoding, and loss agree.
- Parameter count and memory fit the available hardware and latency budget.
- Training-only behavior such as dropout changes correctly at inference.
- Validation data guides architecture choices, while the final test set stays held out.

Architecture is the set of decisions that turns an input shape into an output shape through learnable operations. Once those dimensions, activations, and objectives agree, training has a coherent system to optimize.
