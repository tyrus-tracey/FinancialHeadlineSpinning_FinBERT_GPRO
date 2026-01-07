A completed assignment from https://github.com/tristanengst/fintechCS9832025fa_a3

## Overview
With financial headlines being an information source for market prediction, there is a lot of power in being able to subtly twist headlines to create a narrative. For example, an investor prefers a bullish market to increase their returns so they will want headlines with positive sentiment to encourage investment:

**ORIGINAL:** *"Aspocomp has a large factory in China and a factory building project in India that was halted due to financing problems."*

**ALTERED:** *"Aspocomp unveils a new strategy in manufacturing operations after re-evaluation of priorities."*

LLMs can be trained to do this spinning, and understanding how to setup, implement, and tune the training process was the goal of this assignment.

NOTE: Those in the know will point out *"FinBERT isn't an LLM!"* which is true, but this was chosen for its small footprint.

For more information on Group Relative Policy Optimization (GRPO) which was used to train the model, refer to the [original DeepSeek paper](https://arxiv.org/pdf/2402.03300).

## Run Instructions
For environment setup please refer to the README in the original assignment repo.
To run the training process with the default configuration, simply run `python3 TrainGRPO.py`. Details on hyperparameter configration can be found in the `get_args()` function of TrainGRPO.py, or you can also refer to the best configuration I found located under `Best run config.txt`.
