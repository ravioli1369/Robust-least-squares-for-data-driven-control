function [x_opt, Y_opt, lambda_opt, storage] = grlsq(problem, params)

    Y_hat = params.Y_hat;
    rho = params.rho;
    tolx = params.tolx;
    Sp = params.Sp;
    alpha0 = params.alpha0;
    delta = params.delta;
    gamma = 1e-2;

    options = optimset('Display','off');

    if(strcmp(params.metric, 'gap'))
        dist = @(Y1,Y2) dinf(Y1,Y2);
    elseif(strcmp(params.metric, 'chordal'))
        dist = @(Y1,Y2) d2(Y1, Y2);
    else
        warning('Improper distance metric for Grassmannian');
    end

    x0 = Y_hat*Y_hat'*problem.x0;
    b = problem.b;
    d = problem.d;

    n = size(Y_hat,1);
    k = size(Y_hat,2);

    M = Sp*(Y_hat*Y_hat'); % Linear constraint matrix (M = Sp Y Y' may not have full row rank).
    Hs2 = pinv(M*M');

    x = x0 - M'*Hs2*(M*x0 - d);

    N = 1;
    cost_val = [];
    gradnorm = [];
    iter = [];

    while N <= params.max_iter
        A = x*x' - x*b' - b*x' ;
        [Y,~] = eigs(A,k);
        Y = real(Y);
        lambda_opt = params.lambda0;

        if(dist(Y,Y_hat)<=rho)
            Y_opt = Y;
            lambda_opt = 0;
        else
            % params.x = x;
            % params.b = problem.b;
            % params.dist = dist;
            % lambda_opt = fzero(@(l) ball(l, params), [0, params.lambda0], options);
            while(dist(Y,Y_hat) > rho)
                lambda_opt = lambda_opt*delta;
                H = [sqrt(lambda_opt)*Y_hat, (x-b), 1j*b];
                [U,~,~] = svd(H, 'econ');
                Y = real(U(:,1:k));
            end
            params.lambda0 = lambda_opt/(delta^2);
            % lambda_opt = lambda_opt/delta;
            H = [sqrt(lambda_opt)*Y_hat, (x-b), 1j*b];
            [U,~,~] = svd(H, 'econ');
            Y_opt = real(U(:,1:k));
        end
        if(params.hess_info)
            Q = hessian(Y_opt) + params.eps*eye(n);
        else
            Q = eye(n);
        end
        
        M = Sp*(Y_opt*Y_opt'); % Linear constraint matrix (M = Sp Y Y' may not have full row rank).
        Hs = pinv(M*(Q\M')); % Pre-calculating pseudoinverse
    
        f = @(x) cost(problem, params, x, Y_opt);
        grad_f = @(x) gradf(problem, params, x, Y_opt);
        v = grad_f(x);
        proj_grad = @(x) Q\grad_f(x) - Q\(M'*(Hs*(M*(Q\grad_f(x)))));
        z = Q\v - Q\(M'*(Hs*(M*(Q\v))));
        alpha = backtracking(f, proj_grad, x, z, alpha0);

        if(norm(z)<tolx)
            x_opt = x;
            break
        else
            x = x - alpha*z;
        end
        N = N + 1;


        if(params.store)
            cost_val = [cost_val, f(x)];
            gradnorm = [gradnorm, norm(z)];
            iter = [iter, N];
        end
    end

    storage.cost = cost_val;
    storage.gradnorm = gradnorm;
    storage.iter = iter;


