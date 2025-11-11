oldpath = path;
path('utils/',oldpath)
setup();
rng(42);

%% System Parameters
system.x0 = [0; 0];
mass = 1;               
dt = 0.5;              
system.A = [1, dt; 0, 1];
system.B = [0.5 * dt^2 / mass; dt / mass];
system.C = [1, 0];
system.D = 0;
q = size(system.B,2) + size(system.C,1); 

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
params.rho = sin(2*pi/180);
params.metric = 'chordal';
params.lambda0 = 1e-4;
params.gamma = 4;
params.delta = 1.5;
params.hess_info = true;
params.store = true;

%% Tracking Parameters
r = 1;
T_ini = 10;             % Past window length
N = 25;                 % Prediction horizon
L = T_ini + N;          % Hankel matrix depth
T_d = 115;              % Data length for Hankel
T_sim = 100;             % Simulation steps
params.M = [eye(q*T_ini), zeros(q*T_ini, q*N)];

%% Offline subspace estimation
[params.Y_hat, rhos] = subspace_estimate(system, L, T_d);
% params.Y_hat = Y_opt_wc;

%% Online LQT

% Initialize past data from simulation
x_current = [0; 0];
u_past = zeros(T_ini, 1);
y_past = zeros(T_ini, 1);
for i = 1:T_ini
    u_temp = 0.5 * randn(1);
    x_current = system.A * x_current + system.B * u_temp;
    y_temp = system.C * x_current;
    u_past(i) = u_temp;
    y_past(i) = y_temp;
end
% x_current

% Closed-loop simulation storage
u_cl = zeros(1, T_sim);
y_cl = zeros(1, T_sim);
lambdas = zeros(1,T_sim);
thetas = zeros(1,T_sim);

tic
for t=1:T_sim
    w_ini = [u_past'; y_past'];
    w_ini = w_ini(:);

    % Build reference trajectory
    w_ref = repmat([0; r], [N, 1]);
    problem.b = [w_ini; w_ref];  % Complete reference

    % problem.x0 = zeros(q*L,1);
    problem.x0 = problem.b;

    if(t==50)
        params.store = true;
    else
        params.store = false;
    end
 
    [x_opt, Y_opt, lambda_opt, storage] = grlsq_reg(problem, params);
    % [x_opt, storage] = lsq_reg(problem,params);
    % Y_opt = params.Y_hat;
    % C = params.M*Y_opt*Y_opt';
    % x_opt = x_opt - C'*pinv(C*C')*(C*x_opt - w_ini);

    if(t==50)
        cost = storage.cost;
        gradnorm = storage.gradnorm;
        iter = storage.iter;
    end
    % thetas(1,t) = dinf(Y_opt, params.Y_hat);

    w_opt = Y_opt*(Y_opt'*x_opt);
    w_f_star = w_opt(q*T_ini+1:end); % Future components

    u_opt = w_f_star(1);  % First predicted control input

    % Apply control to system
    x_current = system.A * x_current + system.B * u_opt;
    y_opt = system.C * x_current + system.D * u_opt + randn(1)*s_noise; 
    
    % Update past data (shift window)
    u_past = [u_past(2:end); u_opt];
    y_past = [y_past(2:end); y_opt];
    
    % Store results
    u_cl(t) = u_opt;
    y_cl(t) = y_opt;
    lambdas(t) = lambda_opt;

    fprintf('Step t = %d/%d\n', t, T_sim);

end
toc

err = s_noise*r;
y_lower = y_cl - err;
y_upper = y_cl + err;

% Plot results
figure;
subplot(2, 1, 1);
plot(dt*(1:T_sim), y_cl, 'b', 'LineWidth', 1.5, 'DisplayName','Optimal output $y^*(t)$');
hold on;
patch([dt*(1:T_sim), fliplr(dt*(1:T_sim))], [y_lower, fliplr(y_upper)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
grid on
plot(dt*[1, T_sim], [r, r], 'r--', 'LineWidth', 1.5, 'DisplayName','Reference $r(t)$');
xlabel('Time');
ylabel('Output');
ylim([-3 3]);
legend('Interpreter','latex');
title('Closed-Loop Output');

subplot(2, 1, 2);
plot(dt*(1:T_sim), u_cl, 'g', 'LineWidth', 1.5, 'DisplayName','$u^*(t)$');
hold on
grid on
xlabel('Time');
ylabel('Control Input');
legend('Interpreter','latex')
title('Control Inputs');

% subplot(3, 1, 3);
% plot(dt*(1:T_sim), lambdas, 'k', 'LineWidth', 1.5, 'DisplayName','$\lambda^*(t)$');
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