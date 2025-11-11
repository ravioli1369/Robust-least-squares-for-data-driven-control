function alpha_star = backtracking(f, grad_f, x, v, alpha0, gamma, c)

    if nargin < 5, alpha0 = 0.5; end
    if nargin < 6, gamma = 0.5; end
    if nargin < 7, c = 1e-4; end

    alpha = alpha0;
    fx = f(x);
    grad_fx = grad_f(x);

    % Armijo condition: f(x + alpha*p) <= f(x) + c*alpha*grad_f(x)'*p
    while f(x - alpha * v) > fx - c * alpha * (grad_fx' * v)
        alpha = gamma * alpha;
        if(alpha < 1e-8)
            break
        end
    end
    alpha_star = alpha;
end