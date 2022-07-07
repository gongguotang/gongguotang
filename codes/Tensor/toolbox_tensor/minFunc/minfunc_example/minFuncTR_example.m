clear all

% Make some data
nInstances = 500;
nVars = 50;
X = randn(nInstances,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInstances,1));
flipPos = rand(nInstances,1) < .1;
y(flipPos) = -y(flipPos);

%% Exact Hessian Variants

fprintf('****************************\n** Exact Hessian Variants **\n****************************\n');

options = [];

fprintf('Using Cauchy point\n');
options.solver = 'cauchy';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using piecewise-linear dog-leg path\n');
options.solver = 'dogleg';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using Steihaug CG method\n');
options.solver = 'steihaug';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Solving exactly with binary search\n');
options.solver = 'schur';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Solving in the Linf-norm with QP');
options.solver = 'Linf';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

if exist('minConf_SPG') == 2
    fprintf('Solving exactly with SPG\n');
    options.solver = 'SPG';
    minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
    pause
end

fprintf('***************************\n** Quasi-Newton Variants **\n***************************\n');

fprintf('Using Cauchy w/ BFGS\n');
options.solver = 'cauchy';
options.Hessian = 'bfgs';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using dogleg w/ BFGS\n');
options.solver = 'dogleg';
options.Hessian = 'bfgs';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using Schuur w/ BFGS\n');
options.solver = 'schur';
options.Hessian = 'bfgs';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using Schuur w/ SR1\n');
options.solver = 'schur';
options.Hessian = 'sr1';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using Schuur w/ L-BFGS\n');
options.solver = 'schur';
options.Hessian = 'lbfgs';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Using Schuur w/ L-SR1\n');
options.solver = 'schur';
options.Hessian = 'lsr1';
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('***************************\n** Hessian-Free Variants **\n***************************\n');

fprintf('Using Steihaug CG method\n');
options.solver = 'steihaug';
options.cgSolve = 1;
options.HvFunc = @LogisticHv;
minFuncTR(@LogisticLoss,zeros(nVars,1),options,X,y);
pause