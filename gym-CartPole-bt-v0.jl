"""
This is a Julia version of the following Open AI Gym custom
environment: https://github.com/billtubbs/gym-CartPole-bt-v0/
This version of the classic cart-pole or cart-and-pendulum
control problem offers more variations on the basic OpenAI
Gym version (CartPole-v1).
It is based on a MATLAB implementation by Steven L. Brunton
as part of his Control Bootcamp series of videos on YouTube.
Features of this version include:
- More challenging control objectives (e.g. to stabilize
  the cart x-position as well as the pendulum angle)
- Continuously varying control actions
- Random disturbance to the state
- Measurement noise
- Reduced set of measured state variables
The goal of building this environment was to test different
control engineering and reinfircement learning methods on
a problem that is more challenging than the simple cart-pole
environment provided by OpenAI but still simple enough to
understand and use to help us learn about the relative
strengths and weaknesses of control/RL approaches.
"""

using Printf
using Test
using DifferentialEquations


struct CartPoleBTEnv
    gravity::Float64
    masscart::Float64
    masspole::Float64
    length::Float64
    friction::Float64
    max_force::Float64
    goal_state::Array
    initial_state::String
    disturbances::String
    initial_state_variance::String
    measurement_error::String
    hidden_states::Bool
    variance_levels::Dict
    tau::Float64
    n_steps::Int
    time_step::Int
    kinematics_integrator::String
    observation_space::Array
    action_space::Array
    seed::Int
    state::Array
end

# Set defaults with keyword arguments
CartPoleBTEnv(;
    gravity=-10.0, 
    masscart=5.0, 
    masspole=1.0, 
    length=2.0, 
    friction=1.0, 
    max_force=200.0,
    goal_state=[0.0; 0.0; pi; 0.0],
    initial_state="goal",
    disturbances="none",
    initial_state_variance="none",
    measurement_error="none",  # Not implemented yet
    hidden_states=false,  # Not implemented yet
    variance_levels=Dict("none"=>0.0, "low"=>0.01, "high"=>0.2),
    tau=0.05,
    n_steps=100,
    time_step=0,
    kinematics_integrator="RK45",
    observation_space=[[-Inf64; -Inf64; -Inf64; -Inf64],
                       [Inf64; Inf64; Inf64; Inf64]],
    action_space=[[-Inf64; -Inf64]],
    seed=1,
    state=zeros(4)
) = CartPoleBTEnv(
    gravity, masscart, masspole, length, friction, max_force,
    goal_state, initial_state, disturbances, initial_state_variance,
    measurement_error, hidden_states, variance_levels, 
    tau, n_steps, time_step, kinematics_integrator,
    observation_space, action_space, seed, state
)

# Usage:
# CartPoleBTEnv()  # Defaults
# CartPoleBTEnv(;friction=2)  # Specify non-default values

gym = CartPoleBTEnv()
@test gym.friction == 1.0
@test gym.state == [0.0; 0.0; 0.0; 0.0]
@test gym.goal_state == [0.0; 0.0; 3.141592653589793; 0.0]
