function [mval,g]=Cost_Tensor_Completion_Complex(x,E,T,lamda)
    [n1,n2,n3]=size(T); 
    r=length(x)/2/(n1+n2+n3);
    x = reshape(x,length(x)/2,2);
    x = x(:,1)+1i*x(:,2);
    U = reshape(x(1:n1*r),n1,r);
    V = reshape(x(n1*r+1:(n1+n2)*r),n2,r);
    W = reshape(x((n1+n2)*r+1:end),n3,r);
    dev = cp(ones(r,1),U,V,W).*E - T;
    %% cost
    mval=1/3*lamda*(sum(norms(U).^3)+sum(norms(V).^3)+sum(norms(W).^3))...
        +norm(dev(:))^2/2;   
    %% gradient
    g_U=reshape(permute(dev.*E,[1 3 2]),n1,n3*n2)*conj(kr(V,W))+lamda*U*diag(norms(U));
    g_V=reshape(permute(dev.*E,[2 1 3]),n2,n1*n3)*conj(kr(W,U))+lamda*V*diag(norms(V));
    g_W=reshape(permute(dev.*E,[3 2 1]),n3,n2*n1)*conj(kr(U,V))+lamda*W*diag(norms(W));
    
    g=[real(g_U(:));real(g_V(:));real(g_W(:));imag(g_U(:));imag(g_V(:));imag(g_W(:))];
end