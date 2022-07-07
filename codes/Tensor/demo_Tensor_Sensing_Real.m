% Script to perform overcomplete tensor sensing recover by tensor nuclear norm
% regularized lasso:  min_T ||T-A(T)||_F^2 +lam ||T||_*  
%
% minfunc required https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
% For more information see the paper "Overcomplete Tensor Decomposition via
% Convex Optimization" by Qiuwei Li, Ashley Prater, Lixin Shen, Gongguo Tang
clear;clc;
addpath(genpath('toolbox_tensor/'));

% paramters for minfunc
opts = struct('MAXFUNEVALS',50000,'MAXITER',50000,'display','off','Method',...
    'lbfgs', 'progTol',1e-8, 'optTol',1e-8);

% problem dimenstions and true factors
n1=5;  n2=5; n3=5; r=2;  
U0 = sort1(randn(n1,r)+0i*randn(n1,r));  
V0 = sort1(randn(n2,r)+0i*randn(n2,r)); 
W0 = sort1(randn(n3,r)+0i*randn(n3,r));
lam=(randn(r,1).^2+1)/2;
T = cp(lam,U0,V0,W0);

% tensor sensing 
m=4*(n1+n2+n3)*r; 
A=randn(m,n1*n2*n3);
z=A*T(:);

% random initilizations
U = sort1(randn(n1,r)+0i*randn(n1,r));  
V = sort1(randn(n2,r)+0i*randn(n2,r)); 
W = sort1(randn(n3,r)+0i*randn(n3,r));
lamda=0.000001;
x = [U(:);V(:);W(:)];
%x = [real(x);imag(x)];
 
% solve by minFunc
x = minFunc(@(x)Cost_Tensor_Sensing(x,z,A,lamda,n1,n2,n3),x,opts); 
check_Hessian(@(x)Cost_Tensor_Sensing(x,z,A,lamda,n1,n2,n3),x,1)
% x = reshape(x,(n1+n2+n3)*r,2);
% x = x(:,1)+1i*x(:,2);
U = sort1(reshape(x(1:n1*r),n1,r));
V = sort1(reshape(x(n1*r+1:(n1+n2)*r),n2,r));
W = sort1(reshape(x((n1+n2)*r+1:end),n3,r));
error=norm(U-U0)+norm(V-V0)+norm(W-W0)

 