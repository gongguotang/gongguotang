function [u1,u2,u3] = T_power(T, mxitr)
[n1,n2,n3]=size(T);
uinit=randn(n1,1);u1=uinit/norm(uinit);
uinit=randn(n2,1);u2=uinit/norm(uinit);
uinit=randn(n3,1);u3=uinit/norm(uinit);
for itr=1:mxitr
    v1=zeros(n1,1);v2=zeros(n2,1);v3=zeros(n3,1);
    for i3=1:n3
        v3(i3)=u1'*T(:, :, i3)*u2;
        v1 = v1 + u3(i3)*T(:, :, i3)*u2;
        v2 = v2 + u3(i3)*T(:, :, i3)'*u1;
    end
    u10 = u1;
    u1 = v1/norm(v1);
    u20 = u2;
    u2 = v2/norm(v2);
    u30 = u3;
    u3 = v3/norm(v3);
    if(norm(u10-u1)+norm(u20-u2)+norm(u30-u3)<1e-16) 
    break; 
    end
end
end
