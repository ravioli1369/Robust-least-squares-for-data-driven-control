function z = proj_gradf(problem, x,Y)
    v = gradf(problem, x, Y);
    z = Q\v - Q\(M'*(Hs*(M*(Q\v))));