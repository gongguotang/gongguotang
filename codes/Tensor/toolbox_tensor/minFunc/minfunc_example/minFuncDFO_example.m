clear all

% Make some data
nInstances = 500;
nVars = 50;
X = randn(nInstances,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInstances,1));
flipPos = rand(nInstances,1) < .1;
y(flipPos) = -y(flipPos);


fprintf('Running fminsearch for comparison...\n');
options.Display = 'iter';
options.MaxIter = 2000;
options.MaxFunEvals = 2000;
fminsearch(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with random search...\n');
options.solver = 'random';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with interpolation method...\n');
options.solver = 'interpModel';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with conjugate directions...\n');
options.solver = 'conjugateDirection';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with Hooke Jeeves...\n');
options.solver = 'hookeJeeves';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with coordinate search...\n');
options.solver = 'coordinateSearch';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with pattern search...\n');
options.solver = 'patternSearch';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFuncDFO with Nelder Mead...\n');
options.solver = 'nelderMead';
minFuncDFO(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

fprintf('Running minFunc w/ numerical differencing\n');
options.numDiff = 1;
minFunc(@LogisticLoss,zeros(nVars,1),options,X,y);
pause

