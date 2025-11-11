function g = gradf_reg(problem, params, x,Y)

    M = params.M;
    gamma = params.gamma;
    mu = params.mu;
    b = problem.b;
    g = 2*Y*(Y'*(Y*(Y'*x)-b)) + 2*gamma*Y*(Y'*(M'*((M*(Y*Y'*x - b))))) + 2*mu*x;
    % g = 2*(1+gamma)*Y*(Y'*(x-b));