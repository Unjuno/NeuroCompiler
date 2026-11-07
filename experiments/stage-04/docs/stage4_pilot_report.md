# Stage-4 Semantic-Descriptor Pilot

## Question
Can the compiler replace primitive IDs with semantic descriptors and extrapolate to primitives never active during training?

## Train / held-out primitives
Training uses sin/cos frequencies 1-2 and Legendre degrees 1-5. Held out entirely from training: sin/cos frequency 3 and Legendre degree 6. Each token uses `[is_sin, is_cos, is_legendre, normalized_order]` plus its coefficient.

## Pilot status
One development seed only; not confirmatory. Semantic calibration matching was excellent (log-gap 0.001042), but Meta calibration zero MSE was 1.2389, indicating a weak zero-shot initialization.

For held-out `sin(3*pi*x)`, Semantic DirectFit-final zero-step MSE was 1.8788; Semantic MetaInit reduced its MSE to 0.5012 after 100 adaptation steps.

## Decision
**UNCERTAIN / development failure for zero-shot semantic extrapolation.** The semantic descriptor does not yet generate accurate completed weights for unseen primitives. There is an adaptation signal, but the final-task initial losses are not matched, so this pilot is insufficient to claim optimization-friendliness on unseen primitive identity. A new confirmatory family with fresh held-out primitive orders is required after calibration-only hyperparameter tuning.
