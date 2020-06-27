# gym-julia

This is an alternative version of my [cart pole custom AI Gym environment](https://github.com/billtubbs/gym-CartPole-bt-v0) written in Julia to compare execution speed with the Python version.

- [Test-Julia-version.ipynb](Test-Julia-version.ipynb) - test script and results
- [CartPoleBTEnv.jl](CartPoleBTEnv.jl) - Julia code for simulating environment
- [cartpend.jl](cartpend.jl) - Cart-pendulum dynamics equations

## Initial Test Results

| Test                                   | Python      | Julia         | Ratio      |
|----------------------------------------|-------------|---------------|------------|
| 100 episodes using LSODA solver        | 6.9 seconds |   1.5 seconds |       ~4.5 |
| 100 episodes using Euler approximation | 1.8 seconds | 0.068 seconds |        ~26 |

Test machine: Mac Mini 2012 with 16 GB RAM
