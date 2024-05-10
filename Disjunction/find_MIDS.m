function [MIDS, suma, check, opt] = find_MIDS(A,w)

n = size(A, 1);
I = eye(n);
O = ones(n, 1);

% Solution based on pseudo-inverse
MIDS = (pinv(A+I)*O);
check= (A+I)*MIDS;
suma = sum(MIDS);

% Solution based on optimization
f = sum(A, 1) + 1;
b = -1 * ones(n, 1);
LB = zeros(n, 1);
UB = 1.01 * ones(n, 1);
intcon = 1:n;
Aeq=[];
Beq=[];

opt = intlinprog(f,intcon,-(A+I)*w,b,Aeq,Beq,LB,UB);

end