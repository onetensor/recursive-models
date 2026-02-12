This repo contains some of my (work in progress) research ideas based off of TRM.  

## VRM: Axial Token-Mixing to Replace Dense LxL Mixing

Reshape tokens from `[B, L, D]` into `[B, H, W, D]` (H=W=9 for Sudoku, H=W=30 for Maze/ARC).

Then do two token-mixing steps:

1. **Row-mix:** mix across W within each row (shared weights across rows).
2. **Col-mix:** mix across H within each column (shared weights across cols).

This is the same spirit as token-mixing MLP, but parameterizes it as (essentially) `O(W^2 + H^2)` instead of `O((HW)^2)`.

For 30x30:

- Dense mixing params scale like 900^2 (very expensive).
- Axial mixing params scale like 30^2 + 30^2 = 1800 (okay).

### Pseudocode Sketch

- `X = X + RowMixer(X)` where `RowMixer` applies an MLP/linear map over the width dimension for each row.
- `X = X + ColMixer(X)` similarly over height dimension for each column.

You can implement RowMixer/ColMixer either as:

- Pure linear mixing: `W x W` 
- MLP mixing: `W -> r -> W`

There is also a variant with a depthwise 2d convolution added, which performs similarly well on Sudoku.


On Sudoku-Extreme, VRM achieves SOTA results, with a validation accuracy of 95%, and an exact accuracy (samples that are 100% correct) of 87%, compared to TRM, which reaches 87% and 64% respectively.

<img width="400" height="380" alt="image" src="https://github.com/user-attachments/assets/95b7f412-1bdd-4315-92a7-1ac9a813698d" />
<img width="400" height="380" alt="image" src="https://github.com/user-attachments/assets/bd58bfea-8442-4dab-8736-c1408c0bbf6c" />

Additionally, it matches the performance of TRM in roughly 14k epochs. No hyperparameter sweeps were done, this was with the default hyperparameters within the TRM repo.
\
\
\
\
\
(Other research ideas have not yet been completed or trained)
\
\
\
\
\
This repo was based on: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
