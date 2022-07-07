function V0= T_good(TE,r)
%% Modified by QW %%
ninit=5;
nit=40;
[n1,n2,n3] = size(TE);
% initialization by Robust Tensor Power Method (modified for non-symmetric tensors)
U01 = zeros(n1,r);
U02 = zeros(n2,r);
U03 = zeros(n3,r);
S0 = zeros(r,1);
for i=1:r
    tU1 = zeros(n1,ninit);
    tU2 = zeros(n2,ninit);
    tU3 = zeros(n3,ninit);
    tS  = zeros(ninit,1);
    T_res=TE-cp(S0,U01,U02,U03);
    for init=1:ninit
        [tU1(:,init),tU2(:,init),tU3(:,init)] = T_power(T_res, nit);
        tU1(:,init) = tU1(:,init)./norm(tU1(:,init));
        tU2(:,init) = tU2(:,init)./norm(tU2(:,init));
        tU3(:,init) = tU3(:,init)./norm(tU3(:,init));
        tS(init) = tk(T_res,tU1(:,init),tU2(:,init),tU3(:,init) );
    end
    [~,I] = max(tS);
    U01(:,i) = tU1(:,I)/norm(tU1(:,I));
    U02(:,i) = tU2(:,I)/norm(tU2(:,I));
    U03(:,i) = tU3(:,I)/norm(tU3(:,I));
    T_res_new=TE-cp(S0,U01,U02,U03);
    S0(i) =tk(T_res_new,U01(:,i),U02(:,i),U03(:,i));
end

V0{1}=sort1(U01);V0{2}=sort1(U02);V0{3}=sort1(U03);V0{4}=S0;
end





%% Original %%%
% 
% ninit=20;
% nitr=40;
% [n1,n2,n3] = size(TE);
% % initialization by Robust Tensor Power Method (modified for non-symmetric tensors)
% U01 = zeros(n1,r);
% U02 = zeros(n2,r);
% U03 = zeros(n3,r);
% S0 = zeros(r,1);
% for i=1:r
%     tU1 = zeros(n1,ninit);
%     tU2 = zeros(n2,ninit);
%     tU3 = zeros(n3,ninit);
%     tS  = zeros(ninit,1);
%     for init=1:ninit
%         [tU1(:,init),tU2(:,init),tU3(:,init)] = RTPM(TE-CPcomp(S0,U01,U02,U03), nitr);
%         tU1(:,init) = tU1(:,init)./norm(tU1(:,init));
%         tU2(:,init) = tU2(:,init)./norm(tU2(:,init));
%         tU3(:,init) = tU3(:,init)./norm(tU3(:,init));
%         tS(init) = TenProj(TE-CPcomp(S0,U01,U02,U03),tU1(:,init),tU2(:,init),tU3(:,init) );
%     end
%     [~,I] = max(tS);
%     U01(:,i) = tU1(:,I)/norm(tU1(:,I));
%     U02(:,i) = tU2(:,I)/norm(tU2(:,I));
%     U03(:,i) = tU3(:,I)/norm(tU3(:,I));
%     S0(i) = TenProj(TE-CPcomp(S0,U01,U02,U03),U01(:,i),U02(:,i),U03(:,i));
% end
% 
% V1 = U01;V2 = U02;V3 = U03;
% S = S0;
% V0{1}=sort1(V1);V0{2}=sort1(V2);V0{3}=sort1(V3);V0{4}=sort1(S);
% end
% 
% function T = CPcomp(S,U1,U2,U3)
% n1 = size(U1,1);n2 = size(U2,1);n3 = size(U3,1);
% T = zeros(n1,n2,n3);
% for i=1:n3
%     T(:,:,i) = U1*diag(U3(i,:).*S(:)')*U2';
% end
% end
% 
% function M = TenProj(T,U1,U2,U3)
% n1=size(U1,1);n2=size(U2,1);n3=size(U3,1);
% r1=size(U1,2);r2=size(U2,2);r3=size(U3,2);
% M =zeros(r1,r2,r3);
% for i=1:r3
%     A = zeros(n1,n2);
%     for j=1:n3
%         A = A+T(:,:,j)*U3(j,i);
%     end
%     M(:,:,i) = U1'*A*U2;
% end
% end
% 
% function [u1,u2,u3] = RTPM(T, mxitr)
% n1=size(T, 1);n2=size(T, 2);n3=size(T, 3);
% uinit=randn(n1,1);u1=uinit/norm(uinit);
% uinit=randn(n2,1);u2=uinit/norm(uinit);
% uinit=randn(n3,1);u3=uinit/norm(uinit);
% for itr=1:mxitr
%     v1=zeros(n1,1);v2=zeros(n2,1);v3=zeros(n3,1);
%     for i3=1:n3
%         v3(i3)=u1'*T(:, :, i3)*u2;
%         v1 = v1 + u3(i3)*T(:, :, i3)*u2;
%         v2 = v2 + u3(i3)*T(:, :, i3)'*u1;
%     end
%     u10 = u1;
%     u1 = v1/norm(v1);
%     u20 = u2;
%     u2 = v2/norm(v2);
%     u30 = u3;
%     u3 = v3/norm(v3);
%     if(norm(u10-u1)+norm(u20-u2)+norm(u30-u3)<1e-7) 
%         break; 
%     end
% end
% end
% 
% 
