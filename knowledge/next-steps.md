# Next Steps: rl-values Development

Following the successful integration of Tome and the transition to a training-only architecture, the next phase of development focuses on stability and scaling.

## High-Priority: Training Stability

1.  **Iterative LoRA Refinement**:
    -   [ ] Experiment with smaller LoRA ranks to improve training speed and reduce the size of synchronization updates.
    -   [ ] Investigate the effect of freezing/unfreezing different layers during training (e.g., target only the top half of the transformer).

2.  **Adaptive KL Constraint**:
    -   [ ] Implement a dynamic KL coefficient ($\beta$) that adjusts based on a target KL divergence, preventing the model from collapsing or drifting too far.

3.  **Reward Hacking Monitoring**:
    -   [ ] Build a tool to detect "shortcuts" the model might take (e.g., always giving extremely short responses to avoid nonsense or scrutiny penalties).
    -   [ ] Refine the length penalty criterion to be more robust.

---

## Scaling: Multi-Node Training

1.  **Distributed Tome Nodes**:
    -   [ ] Scale Tome by adding more inference nodes.
    -   [ ] Test the performance of parallel rollouts across multiple GPUs.

2.  **Shared Reward Cache**:
    -   [ ] Centralize the rubric score cache in a shared Redis instance to ensure that identical (prompt, response) pairs are never judged twice across different steps or sessions.

---

## Evaluation & Values

1.  **Rubric Evaluation**:
    -   [ ] Perform formal evaluations of the fine-tuned model against the [Bullshit Bench](https://github.com/petergpt/bullshit-benchmark) to quantify improvements in nonsense detection and scrutiny.
    -   [ ] Conduct human spot-checks to ensure the model maintains intellectual curiosity without becoming overly verbose or evasive.

2.  **Iterative Rubric Refinement**:
    -   [ ] Adjust the weights of each criterion based on early training results.
    -   [ ] Add new criteria if necessary (e.g., "conciseness" or "epistemic humility").
