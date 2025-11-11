function c = lagrangian_reg(problem, params,x,Y,lambda, d)
    b = problem.b;
    M = params.M;
    gamma = params.gamma;
    rho = params.rho;
    mu = params.mu;

    c = norm(Y*(Y'*x) - b)^2+gamma*norm(M*(Y*Y'*x - b))^2+mu*norm(x)^2-lambda*(d^2 - rho^2);
 
   

