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

# TODO: Consider using ModelingToolkit.jl
# https://github.com/SciML/ModelingToolkit.jl

module CartPoleBTEnv

export CartPole, cost_function, step, reset, seed

include("cartpend.jl")

using Printf
using Test
using Random
using DifferentialEquations
using LSODA


mutable struct CartPole
    description::String
    gravity::Float64
    masscart::Float64
    masspole::Float64
    length::Float64
    friction::Float64
    max_force::Float64
    goal_state::Array
    initial_state::Array
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
    rng::AbstractRNG
    state::Array

    # Inner constructor
    function CartPole(;
            # Keyword arguments with default values 
            description::String="Cart-pendulum system",
            goal_state::Array=[0.0; 0.0; pi; 0.0],
            initial_state::Array=[0.0; 0.0; pi; 0.0],
            disturbances::String="none",
            initial_state_variance::String="none",
            measurement_error::String="none",  # Not implemented yet
            hidden_states::Bool=false,  # Not implemented yet
            n_steps::Int=100
        )

        # Physical attributes of system
        gravity::Float64 = -10.0
        masscart::Float64 = 5.0
        masspole::Float64 = 1.0
        length::Float64 = 2.0
        friction::Float64 = 1.0
        max_force::Float64 = 200.0

        # Other features
        variance_levels::Dict = Dict("none"=>0.0, "low"=>0.01, "high"=>0.2)

        # Details of simulation
        tau::Float64 = 0.05
        time_step::Int = 0
        kinematics_integrator::String = "LSODA"
        observation_space::Array = [[-Inf64; -Inf64; -Inf64; -Inf64],
                                    [Inf64; Inf64; Inf64; Inf64]]
        action_space::Array = [-max_force; max_force]
        seed::Int = 1
        rng = MersenneTwister(seed)
        state = zeros(4)  # Initial state is set by reset method
        new(description,
            gravity, masscart, masspole, length, friction, max_force,
            goal_state, initial_state, disturbances, initial_state_variance,
            measurement_error, hidden_states, variance_levels, 
            tau, n_steps, time_step, kinematics_integrator,
            observation_space, action_space, seed, rng, state
        )
    end
end

# Usage:
# CartPole()  # Use defaults
# CartPole(;disturbances="high")  # Specify non-default values


function angle_normalize(theta)
    return theta % (2*pi)
end

function cost_function(state, goal_state)
        """Evaluates the cost based on the current state y and
        the goal state.
        """
        return ((state[1] - goal_state[1])^2 +
                (angle_normalize(state[3]) - goal_state[3])^2)
end
cost_function(gym::CartPole) = cost_function(gym.state, gym.goal_state)
cost_function(gym::CartPole, state) = cost_function(state, gym.goal_state)


# Note: function step already exists in Main so need to 
# extend it as a new method
function Base.step(gym::CartPole, u::Float64)
    u = clamp(u, -gym.max_force, gym.max_force)
    y = gym.state
    t = gym.time_step * gym.tau
    global cost_function
    
    if gym.kinematics_integrator == "Euler"
        y_dot = cartpend_dydt(t, y,
                              gym.masspole,
                              gym.masscart,
                              gym.length,
                              gym.gravity,
                              gym.friction,
                              u)
        gym.state += gym.tau * y_dot
        reward = 0.0  # Not implemented
        done = false  # Not implemented

    else
        # See here https://docs.sciml.ai/v4.0/solvers/ode_solve.html
        if gym.kinematics_integrator == "LSODA"
            # Well-known method which uses switching to solve both 
            # stiff and non-stiff equations
            alg = lsoda()
        elseif gym.kinematics_integrator == "DP5"
            # Dormand-Prince 5/4 Runge-Kutta method
            alg = DP5()
        elseif gym.kinematics_integrator == "BS3"
            # Bogacki-Shampine 3/2 method
            alg = BS3()
        else gym.kinematics_integrator == "Tsit5"
            # Tsitouras 5/4 Runge-Kutta method (default)
            alg = Tsit5()
        end
        f(y, p, t) = cartpend_dydt(t, y,
                                   gym.masspole,
                                   gym.masscart,
                                   gym.length,
                                   gym.gravity,
                                   gym.friction,
                                   u)
        y0 = gym.state
        tspan = (t, t + gym.tau)
        prob = ODEProblem(f, y0, tspan)
        sol = solve(prob, alg)
        gym.state = sol.u[end]
        
    end

    # Add disturbance only to pendulum angular velocity (theta_dot)
    if gym.disturbances != "none"
        v = gym.variance_levels[gym.disturbances]
        gym.state[4] += 0.05 * v * randn(gym.rng)
    end

    reward = -cost_function(gym)
    gym.time_step += 1
    done = (gym.time_step >= gym.n_steps) ? true : false

    return gym.state, reward, done
end


function Base.step(gym::CartPole, u::Array)
    @assert size(u) == (1,)
    step(gym, u[1])
end


function Base.reset(gym::CartPole)
    gym.state = copy(gym.initial_state)
    @assert size(gym.state) == (4,)
    # Add random variance to initial state
    v = gym.variance_levels[gym.initial_state_variance]
    gym.state += v * randn(gym.rng, 4)
    gym.time_step = 0
    return gym.state
end


function seed(gym::CartPole, seed)
    gym.rng = MersenneTwister(seed)
    return seed
end


# Unit Tests
gym = CartPole()
@test gym.friction == 1.0
@test gym.state == [0.0; 0.0; 0.0; 0.0]
@test gym.goal_state == [0.0; 0.0; 3.141592653589793; 0.0]
reset(gym)
@test gym.state == gym.initial_state  # True if variance is none

# Test angle_normalize
@test angle_normalize(0) == 0.0
@test angle_normalize(pi*2.1) == angle_normalize(pi*0.1)
@test angle_normalize(-pi*2.1) == angle_normalize(-pi*0.1)
@test angle_normalize(pi*1.9) == angle_normalize(pi*3.9)

# Test cost_function
@test cost_function(zeros(4), zeros(4)) == 0.0
@test cost_function(zeros(4), [0.0, 0.0, pi, 0.0]) == 9.869604401089358
@test cost_function(gym) == 0.0
@test cost_function(gym, zeros(4)) == 9.869604401089358

# Test step function
gym = CartPole()
gym.state = zeros(4)
state, reward, done = step(gym, 0.0)

end
