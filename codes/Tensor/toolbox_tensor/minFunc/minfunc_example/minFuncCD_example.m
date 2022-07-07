clear all
clear classes

% Make some data
nInstances = 100;
nVars = 200;
X = randn(nInstances,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInstances,1));
flipPos = rand(nInstances,1) < .1;
y(flipPos) = -y(flipPos);

w_init = zeros(nVars,1);

fprintf('Running basic cyclic-Armijo method\n');
options = [];
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Running method with updating\n');
options.updateObj = 1;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Running method with bound for step-size\n');
options.search = -1;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Running method with adaptive bound for step-size\n');
options.search = -2;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Using Wolfe line-search\n');
options.search = 1;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Using Newton line-search\n');
options.search = 2;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Using Newton line-search (quadratic interpolation)\n');
options.search = 2;
options.stepLen = 1;
options.interp = 1;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Using Random Coordinate Selection\n');
options.select = 1;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

fprintf('Sampling According to Lipschitz Constant\n');
tic
 minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause
 
 fprintf('Selecting best value among several random samples\n');
  options.select = 3;
tic
  minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause
  
 fprintf('Greedily selecting best value\n');
options.select = 4;
tic
minFuncCD(LogisticLossClass(X,y,w_init),w_init,options);
toc
pause

