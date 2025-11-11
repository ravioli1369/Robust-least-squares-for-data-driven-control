function [Y_hat, rhos] = subspace_estimate(system, L, T_d)

A = system.A;
B = system.B;
C = system.C;
D = system.D;

m = size(B,2);
p = size(C,1);
n = size(A,1);
q = m+p;

x_sim = zeros(n, T_d);
          
u_d = randn(m, T_d);
y_d = zeros(p, T_d);
y_d_hat = zeros(p, T_d);

% Ratio of signal vs noise magnitude
noise_ratio = 1;

% Magnitude of noise on output measurement and signal
s_noise = 0.2;
s_signal = noise_ratio*s_noise;
x_sim(:, 1) = randn(n,1)*s_signal;

for t = 1:T_d-1
    x_sim(:, t+1) = A * x_sim(:, t) + B * u_d(:,t);
    y_d(:,t) = C*x_sim(:,t) + randn(p)*s_noise;
    y_d_hat(:,t) = C*x_sim(:,t);
end

% Build Hankel matrix
H = zeros(q*L, T_d-L+1);
H_hat = zeros(q*L, T_d-L+1);
for i = 1:(T_d - L + 1)
    col = [];
    col_hat = [];
    for j = 1:L
        col = [col; u_d(:,i+j-1); y_d(:,i+j-1)];
        col_hat = [col_hat; u_d(:,i+j-1); y_d_hat(:,i+j-1)];
    end
    H(:, i) = col;
    H_hat(:,i) = col_hat;
end
% k = rank(H);
k = m*L+n;
% size(H)

if(k>=m*L+n && k<=q*L)
    % SVD for nominal subspace
    [U, ~, ~] = svd(H, 'econ');
    Y_hat = real(U(:, 1:k));
else
    warning('Subspace Estimation failed');
end

rhos = zeros(T_d-L+1,1);

for i=1:T_d-L+1
    w = H(:,i);
    w_hat = H_hat(:,i);
    e = w - w_hat;
    perp = norm((eye(q*L)-Y_hat*Y_hat')*e);
    par = norm(w_hat + Y_hat*(Y_hat'*e));
    rhos(i,1) = atan2(perp,par);
end