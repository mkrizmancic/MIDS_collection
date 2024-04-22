A = [0 1 0 0 1 0;
     1 0 0 0 0 1;
     0 0 0 0 1 0;
     0 0 0 0 1 0;
     1 0 1 1 0 1;
     0 1 0 0 1 0];
I = eye(6);


dD = [0 1 0 0 1 0]';
dJ = [0 1 1 1 0 0]';

JD = (A+I)*dD
JJ = (A+I)*dJ

sumD = sum(JD)
sumJ = sum(JJ)