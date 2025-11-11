function g = gradf(problem, params, x,Y)
    mu = params.mu;
    b = problem.b;
    g = 2*Y*(Y'*(x-b)) + mu*x;