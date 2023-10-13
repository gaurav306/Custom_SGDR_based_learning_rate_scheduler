# Implementation of a custom Stochastic Gradient Descent with Warm Restarts (SGDR) 
This repository includes a Keras callback to be used in training that allows implementation of cyclical learning rate policies, as detailed in Leslie Smith's paper Cyclical Learning Rates for Training Neural Networks arXiv:1506.01186v4.

A cyclical learning rate is a policy of learning rate adjustment that increases the learning rate off a base value in a cyclical nature. Typically the frequency of the cycle is constant, but the amplitude is often scaled dynamically at either each cycle or each mini-batch iteration.

## Args
## Parameters:
- min_lr (float): Minimum learning rate during the entire training.
- max_lr (float): Maximum learning rate before each restart.
- steps_per_epoch (int): Number of steps per epoch. 
                            Is equal to number of training samples divided by batch size. = ceil(num_samples / batch_size)

- first_lr_drop_mult (float): Drop factor for learning rate after the first warmup.
- general_lr_decay (float): Decay factor for learning rate.

- if_warmup_cooldown_start (int): 0 for warmup start, 0 for cooldown start. Decide whether to start with warmup or cooldown step.

- init_cooldown_length (int): Initial number of epochs in a cooldown step.
- init_cooldown_mult_factor (float): Factor to grow the cooldown step length.

- warmup_length (int): Number of epochs in warmup period.
- warmup_mult_factor (float): Factor to grow the warmup length.
- if_no_post_warmup (int): 0 for no post warmup, 1 for post warmup. 
                            Decide whether to have post warmup step, i.e. after first warmup.

- number_of_cooldowns_before_switch (int): Epoch to switch to a new cooldown length.
- new_cooldown_length (int): New cooldown length after switch epoch.
- new_cooldown_mult_factor (float): Factor to grow the new cooldown length.

- verbose (int): Verbosity mode.

## References
  1. **CyclicalLR:** Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/pdf/1506.01186.pdf;
  2. **CosineLR:** SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS: https://arxiv.org/pdf/1608.03983.pdf;

## Examples

### Cosine annealing with initial warmup and decaying max learning rate
```python
DASFSDFG
```

![example1](./TEST/LRS_1.png "example1")
