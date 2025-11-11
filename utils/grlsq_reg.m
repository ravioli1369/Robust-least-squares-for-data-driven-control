function [x_opt, Y_opt, lambda_opt, storage] = grlsq_reg(problem, params)

    Y_hat = params.Y_hat;
    rho = params.rho;
    tolx = params.tolx;
    M = params.M;
    alpha0 = params.alpha0;
    delta = params.delta;
    gamma = params.gamma;

    options = optimset('Display','off');

    if(strcmp(params.metric, 'gap'))
        dist = @(Y1,Y2) dinf(Y1,Y2);
    elseif(strcmp(params.metric, 'chordal'))
        dist = @(Y1,Y2) d2(Y1, Y2);
    else
        warning('Input correct distance metric for Grassmannian');
    end

    x = Y_hat*Y_hat'*problem.x0;
    b = problem.b;

    n = size(Y_hat,1);
    k = size(Y_hat,2);

    M1 = M*(Y_hat*Y_hat'); % Linear constraint matrix (M = Sp Y Y' may not have full row rank).
    Hs2 = pinv(M1*M1');

    x = x - M1'*Hs2*(M1*x - M*b);

    N = 1;
    cost_val = [];
    gradnorm = [];
    iter = [];

    while N <= params.max_iter
        A = x*x' - x*b' - b*x' + gamma*(x*x' - (M'*M)*b*x' - x*b'*(M'*M));
        [Y,~] = eigs(A,k);
        Y = real(Y);
        lambda_opt = params.lambda0;

        if(dist(Y,Y_hat)<=rho)
            Y_opt = Y;
            lambda_opt = 0;
        else
            while(dist(Y,Y_hat) > rho)
                lambda_opt = lambda_opt*delta;
                H = [sqrt(lambda_opt)*Y_hat, (x-b), 1j*b];
                [U,~,~] = svd(H, 'econ');
                Y = real(U(:,1:k));
            end
            params.lambda0 = lambda_opt/(delta^2); % Warm restart

            H = [sqrt(lambda_opt)*Y_hat, (x-b), 1j*b];
            [U,~,~] = svd(H, 'econ');
            Y_opt = real(U(:,1:k));
        end
        if(params.hess_info)
            Q = hessian_reg(Y_opt, params) + params.eps*eye(n);
        else
            Q = eye(n);
        end
        d = dist(Y_opt, Y_hat);
    
        f = @(x) cost_reg(problem, params, x, Y_opt);
        grad_f = @(x) gradf_reg(problem, params, x, Y_opt);
        v = grad_f(x);

        l = @(x) lagrangian_reg(problem, params, x, Y_opt, lambda_opt, d);

        alpha = backtracking(f, grad_f, x, v, alpha0);

        if(norm(v)<tolx)
            x_opt = x;
            break
        else
            x = x - alpha*v;
        end
        N = N + 1;

        if(params.store)
            cost_val = [cost_val, f(x)];
            gradnorm = [gradnorm, norm(v)];
            iter = [iter, N];
        end
    end

    storage.cost = cost_val;
    storage.gradnorm = gradnorm;
    storage.iter = iter;


