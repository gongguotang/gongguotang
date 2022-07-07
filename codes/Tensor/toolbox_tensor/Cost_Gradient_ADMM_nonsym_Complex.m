function [mval,g]=Cost_Gradient_ADMM_nonsym(x,T,y,sigma)
    [n1,n2,n3]=size(T); r=length(x)/(n1+n2+n3);
    U = reshape(x(1:n1*r),n1,r);
    W = reshape(x(n1*r+1:(n1+n2)*r),n2,r);
    Z = reshape(x((n1+n2)*r+1:end),n3,r);
    dev = cp(ones(r,1),U,W,Z) - T;
    %% cost
    mval=1/3*sum(norms(U).^3)+1/3*sum(norms(W).^3)+...
        1/3*sum(norms(Z).^3)-y(:)'*dev(:)+sigma/2*norm(dev(:))^2;   
    %% gradient
    A=sigma*dev-y;
    g_U=reshape(permute(A,[1 3 2]),n1,n3*n2)*kr(W,Z)+U*diag(norms(U));
    g_W=reshape(permute(A,[2 1 3]),n2,n1*n3)*kr(Z,U)+W*diag(norms(W));
    g_Z=reshape(permute(A,[3 2 1]),n3,n2*n1)*kr(U,W)+Z*diag(norms(Z));
    g=[g_U(:); g_W(:); g_Z(:)];
end