"""
Functions to simulate the dynamics of a simple cart-pendulum system.
Copied from MATLAB script provided by Steven L. Brunton as part of his
Control Bootcamp series of YouTube videos.
"""

using Printf
using Test


function cartpend_dydt(t, y, m=1, M=5, L=2, g=-10, d=1, u=0)
    """Simulates the non-linear dynamics of a simple cart-pendulum system.
    These non-linear ordinary differential equations (ODEs) return the
    time-derivative at the current time given the current state of the
    system.
    Args:
        t (float): Time variable - not used here but included for
            compatibility with solvers like scipy.integrate.solve_ivp.
        y (array): State vector. This should be an array of
            shape (4, ) containing the current state of the system.
            y[0] is the x-position of the cart, y[1] is the velocity
            of the cart (dx/dt), y[2] is the angle of the pendulum
            (theta) from the vertical in radians, and y[3] is the
            rate of change of theta (dtheta/dt).
        m (float): Mass of pendulum.
        M (float): Mass of cart.
        L (float): Length of pendulum.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient for friction between cart and
            ground.
        u (float): Force on cart in x-direction.
    Returns:
        dy (array): The time derivate of the state (dy/dt) as a
            shape (4, ) array.
    """

    # Temporary variables
    Sy = sin(y[3])
    Cy = cos(y[3])
    mL = m*L
    D = 1/(L*(M + m*(1 - Cy^2)))
    b = mL*y[4]^2*Sy - d*y[2] + u
    dy = zeros(4)

    # Non-linear ordinary differential equations describing
    # simple cart-pendulum system dynamics
    dy[1] = y[2]
    dy[2] = D*(-mL*g*Cy*Sy + L*b)
    dy[3] = y[4]
    dy[4] = D*((m + M)*g*Sy - Cy*b)

    return dy
end;

function cartpend_ss(m=1, M=5, L=2, g=-10, d=1, s=1)
    """Calculates the linearized approximation of the cart-pendulum
    system dynamics at either the vertical-up position (s=1) or
    vertical-down position (s=-1).

    Returns two arrays, A, B which are the system and input matrices
    in the state-space system of differential equations:

        x_dot = Ax + Bu

    where x is the state vector, u is the control vector and x_dot
    is the time derivative (dx/dt).

    Args:
        m (float): Mass of pendulum.
        M (float): Mass of cart.
        L (float): Length of pendulum.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient for friction between cart and
            ground.
        s (int): 1 for pendulum up position or -1 for down.

    Returns:
        dy (np.array): The time derivate of the state (dy/dt) as a
            shape (4, ) array.
    """

    A = [     0.0        1.0              0.0      0.0;
                0       -d/M           -m*g/M      0.0;
              0.0        0.0              0.0      1.0;
              0.0 -s*d/(M*L) -s*(m+M)*g/(M*L)      0.0]

    B = [        0.0;
               1.0/M;
                 0.0;
         s*1.0/(M*L)]

    return A, B
end;


# Test cartpend_dydt
# Fixed parameter values
m = 1
M = 5
L = 2
g = -10
d = 1
u = 0

y_test_values = Dict(
    1 => [0, 0, 0, 0],  # Pendulum down position
    2 => [0, 0, pi, 0],  # Pendulum up position
    3 => [0, 0, 0, 0],
    4 => [0, 0, pi, 0],
    5 => [2.260914, 0.026066, 0.484470, -0.026480]
);

u_test_values = Dict(
    1 => 0.,
    2 => 0.,
    3 => 1.,
    4 => 1.,
    5 => -0.59601
);

# dy values below calculated with MATLAB script from
# Steven L. Brunton's Control Bootcamp videos
expected_results = Dict(
    1 => [0., 0., 0., 0.],
    2 => [0., -2.44929360e-16, 0., -7.34788079e-16],
    3 => [0., 0.2, 0., -0.1],
    4 => [0., 0.2, 0. ,0.1],
    5 => [0.026066, 0.670896, -0.026480, -2.625542]
);

t = 0.0
atol = 1e-6
for i in 1:5
    u = u_test_values[i]
    y = y_test_values[i]
    dy_calculated = cartpend_dydt(t, y, m, M, L, g, d, u)
    dy_expected = expected_results[i]
    @test maximum(abs.(dy_calculated - expected_results[i])) < atol
end

# Test cartpend_ss
# K values below calculated with MATLAB script from
# Steven L. Brunton's Control Bootcamp videos
test_values = Dict(
    5 => 1,  # Pendulum up position
    6 => -1  # Pendulum down position
)

expected_results = Dict(
    5 => ([0.0   1.0   0.0   0.0;
           0.0  -0.2   2.0   0.0;
           0.0   0.0   0.0   1.0;
           0.0  -0.1   6.0   0.0],
         [ 0.0;  0.2;  0.0;  0.1]),
    6 => ([0.0   1.0   0.0   0.0;
           0.0  -0.2   2.0   0.0;
           0.0   0.0   0.0   1.0;
           0.0   0.1  -6.0   0.0],
         [ 0.0;  0.2;  0.0; -0.1])
);

atol = 1e-6
for i in 5:6
    s = test_values[i]
    A_calculated, B_calculated = cartpend_ss(m, M, L, g, d, s)
    A_expected, B_expected = expected_results[i]
    @test maximum(abs.(A_calculated - A_expected)) < atol
    @test maximum(abs.(B_calculated - B_expected)) < atol
end
