using Test

include("CartPoleBTEnv.jl")
using .CartPoleBTEnv


# Unit Tests for CartPoleBTEnv.jl
env = CartPole()
@test env.friction == 1.0
@test env.state == [0.0; 0.0; 0.0; 0.0]
@test env.goal_state == [0.0; 0.0; 3.141592653589793; 0.0]
reset!(env)
@test env.state == env.initial_state  # True if variance is none

# Test reset and step functions
env = CartPole()
reset!(env)
state, reward, done = step!(env, 0.0)
@test state == env.state
@test env.time_step == 1

# Test random initialization and disturbances
env = CartPole()
env.initial_state_variance = "low"
env.disturbances = "low"
seed = env.seed
reset!(env)
initial_state = env.state
state, reward, done = step!(env, 10.0)
reset!(env)
initial_state2 = env.state
state2, reward2, done2 = step!(env, 10.0)
@test initial_state2 != initial_state
@test state2 != state

# Test repeatability with seeded RNG
seed!(env, seed)
reset!(env)
initial_state3 = env.state
state3, reward3, done3 = step!(env, 10.0)
@test initial_state3 == initial_state
@test state3 == state
@test reward3 == reward
@test done3 == done

# Test step solutions for each solver
solver_names = ["Euler", "LSODA", "DP5", "BS3", "Tsit5"]

# Compare state values after 5 timesteps with those from Python 
# Open AI env implementation (cartpole_bt_env.py)
# with 'RK45' solver: [0.06179183, 0.49282067, 3.17330351, 0.2594884 ]
# With LSODA solver: [0.06183026, 0.49287447, 3.17335713, 0.2596762 ]
# With Euler method: [0.04956447, 0.49255687, 3.166531, 0.2524997 ]
state_test_values = Dict(
    "Euler" => [0.04956447, 0.49255687, 3.166531, 0.2524997 ],
    "LSODA" => [0.06183026, 0.49287447, 3.17335713, 0.2596762 ],
    "DP5" => Array([0.06179183, 0.49282067, 3.17330351, 0.2594884 ]), 
    "BS3" => Array([0.06179183, 0.49282067, 3.17330351, 0.2594884 ]), 
    "Tsit5" => Array([0.06179183, 0.49282067, 3.17330351, 0.2594884 ])
)
# Biggest difference is in state[4] with LSODA method:
# rel_diff = [1.35586e-5, 2.614e-5, 2.79553e-6, 0.000161289]

env = CartPole()
env.disturbances = "none"
env.initial_state_variance = "none"
for name in solver_names
    env.kinematics_integrator = name
    reset!(env)
    for i in 1:5
        state, reward, done = step!(env, 10.0)
    end
    rel_diff = abs.(env.state - state_test_values[name]) ./ env.state
    @test env.time_step == 5
    @test maximum(rel_diff) < 0.0002
end
