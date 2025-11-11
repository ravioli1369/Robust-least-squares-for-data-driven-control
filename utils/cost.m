function c = cost(problem, params, x,Y)
    mu = params.mu;
    b = problem.b;
    c = norm(Y*(Y'*x) - b)^2 + 0.5*mu*norm(x)^2;
   

