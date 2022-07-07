function M = tk(T,U1,U2,U3)
n1=size(U1,1);n2=size(U2,1);n3=size(U3,1);
r1=size(U1,2);r2=size(U2,2);r3=size(U3,2);
M =zeros(r1,r2,r3);
for i=1:r3
    A = zeros(n1,n2);
    for j=1:n3
        A = A+T(:,:,j)*U3(j,i);
    end
    M(:,:,i) = U1'*A*U2;
end
end