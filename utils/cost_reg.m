function c = cost_reg(problem, params,x,Y)
    mu = params.mu;
    b = problem.b;
    M = params.M;
    gamma = params.gamma;

    c = norm(Y*(Y'*x) - b)^2 + gamma*norm(M*(Y*Y'*x - b))^2 + mu*norm(x)^2;
 
   

