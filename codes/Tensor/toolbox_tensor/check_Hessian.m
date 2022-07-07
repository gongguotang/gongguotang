function  check_Hessian(funObj,x,order,varargin)
p = length(x);
d = sign(randn(p,1));
if order == 2
fprintf('Checking Hessian-vector product along random direction:\n');
%% User HV
[~,~,H] = funObj(x,varargin{:});
Hv = H*d; 
%% Numerical HV	
mu = 2*sqrt(1e-12)*(1+norm(x))/(1+norm(x));
[~,diffa] = funObj(x+d*mu,varargin{:});
[~,diffb] = funObj(x-d*mu,varargin{:});
Hv2 = (diffa-diffb)/(2*mu);
fprintf('Max difference : %e\n',max(abs(Hv-Hv2)));
else
fprintf('Checking Gradient along random direction:\n');
%% User directional-derivative
[~,g] = funObj(x,varargin{:});
gtd = g'*d;
%% Numerical directional-derivative	
mu = 2*sqrt(1e-12)*(1+norm(x))/(1+norm(x));
diff1 = funObj(x+d*mu,varargin{:});
diff2 = funObj(x-d*mu,varargin{:});
gtd2 = (diff1-diff2)/(2*mu);
fprintf('Max difference : %e\n',max(abs(gtd-gtd2)));
end

end