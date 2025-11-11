function d = d2(Y1, Y2)
    n = size(Y1,1);
    d = sqrt(trace((eye(n) - Y1*Y1')*(Y2*Y2')));