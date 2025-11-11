function h = hessian_reg(Y, params)
    gamma = params.gamma;
    M = params.M;
    n = size(Y,1);
    mu = params.mu;
    h = 2*(Y*Y') + gamma*2*(Y*Y')*(M'*M)*(Y*Y') + 2*mu*eye(n);