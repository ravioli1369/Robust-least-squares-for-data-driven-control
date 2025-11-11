oldpath = path;
path('utils/',oldpath)
setup();
rng(42);

%% System Parameters
system.x0 = [-1; 1; 0];
% mass = 1;               
dt = 0.01;              
system.A = [[1.01, 0.01, 0]; [0.01, 1.01, 0.01]; [0, 0.01, 1.01]];
system.B = eye(3);
system.C = [1, 0, 0];
system.D = [0, 0, 0];
m = size(system.B,2);
p = size(system.C,1);
q = m + p;

% Ratio of signal vs noise magnitude
noise_ratio = 1;

% Magnitude of noise on output measurement and signal
s_noise = 0.2;
s_signal = noise_ratio*s_noise;

%% Optimization Parameters
params.alpha0 = 0.5;
params.max_iter = 1e5;               
params.tolx = 1e-6;
params.eps = 1e-4; 
params.mu = 0;
params.rho = sin(3.5*pi/180);
params.metric = 'chordal';
params.lambda0 = 1e-4;
params.gamma = 4;
params.delta = 1.5;
params.hess_info = true;
params.store = true;

%% Tracking Parameters
r = 0;
T_ini = 10;             % Past window length
N = 25;                 % Prediction horizon
L = T_ini + N;          % Hankel matrix depth
T_d = 150;              % Data length for Hankel
T_sim = 100;             % Simulation steps
params.M = [eye(q*T_ini), zeros(q*T_ini, q*N)];

%% Offline subspace estimation
[params.Y_hat, rhos] = subspace_estimate(system, L, T_d);

%% Online LQT

% Initialize past data from simulation
x_current = system.x0;
u_past = zeros(T_ini, m);
y_past = zeros(T_ini, p);
for i = 1:T_ini
    u_temp = randn(m,1);
    x_current = system.A * x_current + system.B * u_temp;
    y_temp = system.C * x_current;
    u_past(i,:) = u_temp;
    y_past(i,:) = y_temp;
end

% Closed-loop simulation storage
u_cl = zeros(m, T_sim);
y_cl = zeros(p, T_sim);
lambdas = zeros(1,T_sim);
thetas = zeros(1,T_sim);

tic
for t=1:T_sim
    w_ini = [u_past'; y_past'];
    w_ini = w_ini(:);

    % Build reference trajectory
    w_ref = repmat([zeros(m,1); r], [N, 1]);
    problem.b = [w_ini; w_ref];  % Complete reference

    % problem.x0 = zeros(q*L,1);
    problem.x0 = problem.b;

    if(t==10)
        params.store = true;
    else
        params.store = false;
    end
    [x_opt, Y_opt, lambda_opt, storage] = grlsq_reg(problem, params);
    % [x_opt, storage] = lsq_reg(problem,params);
    % Y_opt = params.Y_hat;

    if(t==10)
        cost = storage.cost;
        gradnorm = storage.gradnorm;
        iter = storage.iter;
    end
    % thetas(1,t) = dinf(Y_opt, params.Y_hat);

    w_opt = Y_opt*(Y_opt'*x_opt);
    w_f_star = w_opt(q*T_ini+1:end); % Future components

    u_opt = w_f_star(1:m);  % First predicted control input

    % Apply control to system
    x_current = system.A * x_current + system.B * u_opt;
    y_opt = system.C * x_current + system.D * u_opt + randn(1)*s_noise; 
    
    % Update past data (shift window)
    u_past = [u_past(2:end,:); u_opt'];
    y_past = [y_past(2:end,:); y_opt'];
    
    % Store results
    u_cl(:,t) = u_opt;
    y_cl(:,t) = y_opt;
    lambdas(t) = lambda_opt;

    fprintf('Step t = %d/%d\n', t, T_sim);

end
toc

% Plot results
figure;
subplot(2, 1, 1);
plot(dt*(1:T_sim), y_cl, 'b', 'LineWidth', 1.5, 'DisplayName','Optimal output $y^*(t)$');
hold on;
grid on
plot(dt*[1, T_sim], [r, r], 'r--', 'LineWidth', 1.5, 'DisplayName','Reference $r(t)$');
xlabel('Time');
ylabel('Output');
ylim([-3 3]);
xlim(dt*[0 T_sim])
legend('Interpreter','latex');
title('Closed-Loop Output');

subplot(2, 1, 2);
plot(dt*(1:T_sim), u_cl(1,:), 'k', 'LineWidth', 1.5, 'Marker','o', 'DisplayName','$u_1^*(t)$');
hold on
grid on
plot(dt*(1:T_sim), u_cl(2,:), 'r', 'LineWidth', 1.5,'Marker','o','DisplayName','$u_2^*(t)$');
plot(dt*(1:T_sim), u_cl(3,:), 'b', 'LineWidth', 1.5, 'Marker','o','DisplayName','$u_3^*(t)$');
xlabel('Time');
xlim(dt*[0 T_sim])
ylabel('Control Input');
legend('Interpreter','latex')
title('Control Inputs');

% subplot(3, 1, 3);
% plot((1:T_sim), lambdas, 'k', 'LineWidth', 1.5, 'DisplayName','$\lambda^*(t)$');
% grid on
% hold on
% xlabel('Time');
% ylabel('Constraint multiplier');
% legend('Interpreter','latex')

figure
subplot(2, 1, 1);
grid on
plot(iter, cost, 'k', 'LineWidth', 1.5, 'DisplayName','Cost $f(x)$');
xlabel('Iterations');
ylabel('Cost');
legend('Interpreter','latex', 'Location','northeast');
title('Cost function versus iteration at t = 5');

subplot(2, 1, 2);
loglog(iter, gradnorm, 'r', 'LineWidth', 1.5, 'DisplayName','Gradnorm $||v||$');
xlabel('Iterations');
grid on
ylabel('Gradnorm');
legend('Interpreter','latex', 'Location','northeast')
title('Norm of Gradient versus iteration at t = 5');

% plot(1:t, params.rho*ones(length(1:t)), 'r--')
% plot(1:t, asin(thetas), 'k', 'DisplayName', '$d_{\infty}(Y_k^*, \hat{Y})$')
% hold on
% plot(1:t, params.rho*ones(length(1:t),1), 'r--', 'DisplayName','$\rho$')
% xlabel('Time, t')
% ylabel('Grassmannian distance')
% legend('Location', 'best', 'Interpreter', 'latex')