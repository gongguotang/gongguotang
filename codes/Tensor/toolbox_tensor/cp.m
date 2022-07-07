function X = cp(varargin)
%% Size of Tensor
sz=zeros(1,nargin-1);
for it=2:nargin
    sz(it-1)=size(varargin{it},1);
end
%% Do Vectorized linear combination
c=varargin{1};
X=varargin(end:-1:2);
X=kr(X);
X=X*c;
%% Reshape
X=reshape(X,sz);
end


