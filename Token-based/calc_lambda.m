function [lambda] = calc_lambda(A)

D = diag(sum(A, 2));
L = D - A;

[~, v] = eig(L);

v = sort(diag(v));

lambda = v(2);
end

