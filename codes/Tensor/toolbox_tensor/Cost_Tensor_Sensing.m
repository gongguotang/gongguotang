function [mval,g]=Cost_Tensor_Sensing(x,z,A,lamda,n1,n2,n3)
    r=length(x)/(n1+n2+n3);
    U = reshape(x(1:n1*r),n1,r);
    V = reshape(x(n1*r+1:(n1+n2)*r),n2,r);
    W = reshape(x((n1+n2)*r+1:end),n3,r);
    T=cp(ones(r,1),U,V,W);
    dev = A*T(:) - z;
    %% cost
    mval=1/3*lamda*(sum(norms(U).^3)+sum(norms(V).^3)+sum(norms(W).^3))...
        +norm(dev)^2/2;   
    %% gradient
g_U=reshape(permute(reshape(A'*dev,n1,n2,n3),[1 3 2]),n1,n3*n2)*conj(kr(V,W))+lamda*U*diag(norms(U));
g_V=reshape(permute(reshape(A'*dev,n1,n2,n3),[2 1 3]),n2,n1*n3)*conj(kr(W,U))+lamda*V*diag(norms(V));
g_W=reshape(permute(reshape(A'*dev,n1,n2,n3),[3 2 1]),n3,n2*n1)*conj(kr(U,V))+lamda*W*diag(norms(W));
g=[g_U(:); g_V(:); g_W(:)];
end