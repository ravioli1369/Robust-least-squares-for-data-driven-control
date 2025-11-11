%% Robust Data-Driven MPC (DeePC) for a Double Integrator
%
% This script implements the robust data-driven model predictive control (DD-MPC)
% scheme based on the paper:
%   "Data-Driven Model Predictive Control With Stability and Robustness Guarantees"
%   by J. Berberich, J. Köhler, M. A. Müller, and F. Allgöwer (2021).
%
% This is the foundational robust QP-based algorithm that the min-max
% approaches (like those by Xie et. al.) are built upon.
%
% PROBLEM: Control a (noisy) double integrator to track a reference signal.

% REQUIRES:
%   1. MATLAB
%   2. YALMIP Toolbox (freely available: https://yalmip.github.io/)
%   3. A QP solver (like 'quadprog' from MATLAB's Optimization Tbx)
%

%% 1. Setup
clear; clc; close all;
rng(42);
fprintf('Robust Data-Driven MPC for Double Integrator...\n');

% Check if YALMIP is installed
if ~exist('sdpvar', 'file')
    error('YALMIP toolbox not found. Please add it to your MATLAB path.');
end

%% 2. Define the "Unknown" System (Double Integrator)
% We use this *only* for simulation. The controller *never* sees A, B, C.
Ts = 0.5; % Sample time
A_c = [0 1; 0 0];
B_c = [0; 1];
C_c = [1 0];
D_c = 0;
sys_c = ss(A_c, B_c, C_c, D_c);
sys_d = c2d(sys_c, Ts, 'zoh');

A_d = sys_d.A;
B_d = sys_d.B;
C_d = sys_d.C;

n = size(A_d, 1); % System order (n=2)
m = size(B_d, 2); % Input dim (m=1)
p = size(C_d, 1); % Output dim (p=1)

%% 3. Collect Data from the "Unknown" System
% We need one persistently exciting trajectory collected offline.
T_data = 115; % Total length of data collection
fprintf('Collecting %d samples of data...\n', T_data);

% Generate a persistently exciting input signal
u_data = 0.5 * randn(m, T_data);

% Add some process noise (zero for comparison)
w_noise_level = 0.0;
w_data = w_noise_level * randn(n, T_data + 1);

% Add some measurement noise
v_noise_level = 0.2;
v_data = v_noise_level * randn(p, T_data);

% Simulate to get noisy data
x_data = zeros(n, T_data + 1);
y_data_noisy = zeros(p, T_data);
x_data(:, 1) = [0; 0];

for k = 1:T_data
    y_data_noisy(:, k) = C_d * x_data(:, k) + v_data(:, k);
    x_data(:, k+1) = A_d * x_data(:, k) + B_d * u_data(:, k) + w_data(:, k);
end
fprintf('Data collection complete.\n');

%% 4. Define MPC and DeePC Parameters
% Horizons
T_ini = 10;  % Past data horizon (initial condition)
N_f = 25;   % Future prediction horizon
L = T_ini + N_f; % Total window length

% Number of columns in Hankel matrices
K = T_data - L + 1;
if K <= 0
    error('Data length T_data is too short for the chosen horizons T_ini and N_f.');
end

% Cost Matrices
Q = 1 * eye(p);
R = 0.5 * eye(m);

% Robustness Regularization Parameters
lambda_g = 0.5;   % Penalty on ||g||_2^2
lambda_s = 1e3; % Penalty on slack ||sigma_y||_2^2

% Constraints
u_min = -Inf;
u_max = Inf;
y_min = -Inf;
y_max = Inf;

%% 5. Construct Hankel Matrices
% This is the core of the data-driven representation
fprintf('Building Hankel matrices...\n');

% Use our local helper function
U_hankel = build_hankel_matrix(u_data, L, K);
Y_hankel = build_hankel_matrix(y_data_noisy, L, K);

% Split into past and future
% Note: T_ini and N_f are in *samples*. We must account for input/output dims.
Up_rows = 1 : (m * T_ini);
Uf_rows = (m * T_ini + 1) : (m * L);
Yp_rows = 1 : (p * T_ini);
Yf_rows = (p * T_ini + 1) : (p * L);

H_p = U_hankel(Up_rows, :);
H_f = U_hankel(Uf_rows, :);
H_y_p = Y_hankel(Yp_rows, :);
H_y_f = Y_hankel(Yf_rows, :);


%% 6. Setup YALMIP Optimization Problem (do this once)
fprintf('Setting up YALMIP optimization problem...\n');

% Define optimization variables
g = sdpvar(K, 1);
sigma_y = sdpvar(p * T_ini, 1);
u_f = sdpvar(m * N_f, 1);
y_f = sdpvar(p * N_f, 1);

% Define parameters (these will be updated in the loop)
u_past_param = sdpvar(m * T_ini, 1);
y_past_param = sdpvar(p * T_ini, 1);
r_f_param = sdpvar(p * N_f, 1); % Future reference vector

% Create block-diagonal cost matrices
Q_bar = kron(eye(N_f), Q);
R_bar = kron(eye(N_f), R);

% Define the Cost Function
J = (y_f - r_f_param)' * Q_bar * (y_f - r_f_param) ... % 1. Tracking cost
  + u_f' * R_bar * u_f ...                          % 2. Input cost
  + lambda_g * (g' * g) ...                         % 3. Regularization on g
  + lambda_s * (sigma_y' * sigma_y);                % 4. Penalty on slack (robustness)

% Define the Constraints
Cns = [u_f == H_f * g];                         % Future input trajectory
Cns = [Cns, y_f == H_y_f * g];                  % Future output trajectory
Cns = [Cns, u_past_param == H_p * g];           % Match past inputs
Cns = [Cns, y_past_param == H_y_p * g + sigma_y]; % Match past outputs (with slack)
Cns = [Cns, u_min <= u_f <= u_max];             % Input constraints
Cns = [Cns, y_min <= y_f <= y_max];             % Output constraints

% Setup the optimizer object
ops = sdpsettings('verbose', 0, 'solver', 'quadprog');
controller = optimizer(Cns, J, ops, ...
                       {u_past_param, y_past_param, r_f_param}, ... % Inputs
                       {u_f, y_f, g, sigma_y});                     % Outputs

fprintf('Controller is ready.\n');

%% 7. Run Simulation
T_sim = 60; % Simulation length
fprintf('Running simulation for %d steps...\n', T_sim);

% Reference trajectory
r_sim = ones(p, T_sim + N_f);

% Storage for simulation results
x_hist = zeros(n, T_sim + 1);
u_hist = zeros(m, T_sim);
y_hist = zeros(p, T_sim);
g_norm_hist = zeros(1, T_sim);
slack_norm_hist = zeros(1, T_sim);

% Initial state and past data buffers
x_hist(:, 1) = [-1.7432; -0.7727]; % True initial state
u_past_buffer = zeros(m, T_ini);
y_past_buffer = zeros(p, T_ini);

x_current = [-1.7432; -0.7727]; % same IC for comparison
% --- Simulation Loop ---

for k = 1:T_sim
    % Get current measurement (from "real" system)
    y_current = C_d * x_current + v_noise_level * randn(p, 1);

    % Update past data buffers
    y_past_buffer = [y_past_buffer(:, 2:end), y_current];
    
    % Prepare data for YALMIP
    % YALMIP needs flat vectors
    u_past_vec = reshape(u_past_buffer, m * T_ini, 1);
    y_past_vec = reshape(y_past_buffer, p * T_ini, 1);
    
    % Get reference for the current horizon
    r_f_vec = reshape(r_sim(:, k : k + N_f - 1), p * N_f, 1);

    % --- Solve the DD-MPC problem ---
    % Pass current data to the pre-compiled optimizer
    [outputs, sol_status] = controller({u_past_vec, y_past_vec, r_f_vec});
    
    if sol_status ~= 0
        warning('Solver failed at step k = %d!', k);
        u_apply = 0; % Fail-safe
        u_f_opt = zeros(m * N_f, 1);
        g_opt = zeros(K, 1);
        sigma_y_opt = zeros(p * T_ini, 1);
    else
        u_f_opt = outputs{1};
        g_opt = outputs{3};
        sigma_y_opt = outputs{4};
        
        % Extract the first control input to apply
        u_apply = u_f_opt(1:m);
    end

    % --- Apply control to the "real" system ---
    w_k = w_noise_level * randn(n, 1); % Process noise
    x_next = A_d * x_current + B_d * u_apply + w_k;

    % --- Store results ---
    x_hist(:, k+1) = x_next;
    y_hist(:, k) = y_current;
    u_hist(:, k) = u_apply;
    g_norm_hist(k) = norm(g_opt);
    slack_norm_hist(k) = norm(sigma_y_opt);
    
    % --- Update for next loop ---
    x_current = x_next;
    u_past_buffer = [u_past_buffer(:, 2:end), u_apply]; % Update input buffer
    
    if mod(k, 10) == 0
        fprintf('Sim step %d/%d\n', k, T_sim);
    end
end
fprintf('Simulation finished.\n');

%% 8. Plot Results
t_vec = (0:T_sim-1) * Ts;

% Plot 1: Output vs Reference
figure;
subplot(2,1,1);
plot(t_vec, y_hist, 'b', 'LineWidth', 1.5);
hold on;
plot(t_vec, r_sim(1:T_sim), 'r--', 'LineWidth', 1.5);
title('Robust Data-Driven MPC (Double Integrator)');
xlabel('Time (s)');
ylabel('Output (y)');
legend('Measured Output', 'Reference');
grid on;

% Plot 2: Control Input
subplot(2,1,2);
plot(t_vec, u_hist, 'k', 'LineWidth', 1.5);
hold on;
plot(t_vec, u_max * ones(1, T_sim), 'r:');
plot(t_vec, u_min * ones(1, T_sim), 'r:');
xlabel('Time (s)');
ylabel('Input (u)');
legend('Control Input', 'Constraints');
grid on;

% Plot 3: Internal variables (for debugging)
figure;
subplot(2,1,1);
plot(t_vec, g_norm_hist, 'm', 'LineWidth', 1.5);
title('Internal Controller Variables');
ylabel('Norm of g ||g||_2', 'Interpreter','latex');
grid on;

subplot(2,1,2);
plot(t_vec, slack_norm_hist, 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Norm of slack ||\sigma_y||_2', 'Interpreter','latex');
grid on;


%% Helper Function to Build Hankel Matrix
function H = build_hankel_matrix(data, L, K)
    % data: (dim, T) data matrix (e.g., u_data)
    % L: Window length (T_ini + N_f)
    % K: Number of columns (T_data - L + 1)
    
    [dim, ~] = size(data);
    H = zeros(dim * L, K);
    
    for j = 1:K % For each column
        % Extract the window
        window = data(:, j : j + L - 1);
        % Reshape and stack it into a column vector
        H(:, j) = reshape(window, dim * L, 1);
    end
end