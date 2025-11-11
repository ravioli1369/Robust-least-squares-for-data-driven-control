function e = ball(l, params)
    Y_hat = params.Y_hat;
    b = params.b;
    x = params.x;
    dist = params.dist;
    rho = params.rho;

    k = size(Y_hat,2);

    % H = [sqrt(l)*Y_hat, (x-b), 1j*b];
    % [U,~,~] = svd(H,'econ');
    B = x*x' - x*b' - b*x' + l*(Y_hat*Y_hat');
    [Y,~] = eigs(B,k);
    Y = real(Y);
    % Y = real(U(:,1:k));

    e = dist(Y,Y_hat) - rho;