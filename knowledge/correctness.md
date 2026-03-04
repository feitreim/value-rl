# Correctness and Parity: rl-values Training

Ensuring that the gradients and logprobs calculated in `rl-values` are accurate is critical for training stability.

## Verification Suite (`tests/`)

We maintain a set of correctness tests to ensure the training path matches expectations.

### 1. Gradient Flow (`test_correctness.py`)
-   **Operation**: Computes the gradient of the GRPO loss with respect to the model parameters.
-   **Success Criterion**: Gradients must be non-zero and flow through the entire transformer (attention and MLP layers).
-   **Verification**: `nn.value_and_grad` is used to confirm differentiability.

### 2. Batch Consistency (`test_correctness.py`)
-   **Operation**: Compares logprobs for the same (prompt, response) pair across different batch sizes ($B=1, 2, 4, 8$).
-   **Success Criterion**: 
    -   **Within-batch**: Logprobs for identical items in the same batch must be identical (diff = 0.00).
    -   **Between-batch**: Logprobs for the same item across different batch sizes should be very close (max diff < 0.50). 
-   **Note**: Minor variance across batch sizes is expected due to the non-deterministic nature of BF16 attention operations on Metal.

### 3. Logprob Parity (`test_tome.py`)
-   **Operation**: Compares the logprobs calculated locally by `group_logprobs` (single forward pass) with those calculated autoregressively by Tome (using a KV cache).
-   **Success Criterion**: Max difference between local and remote logprobs should be small (< 0.50).

---

## Technical Details

### `group_logprobs` (Single Forward Pass)
During the backward pass, `group_logprobs` computes the log-probabilities of a full response in a single forward pass by concatenating the prompt and response.
-   **Efficiency**: Much faster than autoregressive decoding.
-   **Differentiability**: Fully differentiable under the current policy (π_θ).

### Importance Sampling
We use the **Policy Logprobs** from Tome (π_old) as the denominator for the importance sampling ratio:
$\text{ratio} = \exp(\text{logprob}_{\theta} - \text{logprob}_{\text{old}})$
This ensures that the gradient update is correctly weighted relative to the version of the policy that actually generated the rollouts.
