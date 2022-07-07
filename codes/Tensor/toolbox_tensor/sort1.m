function V1=sort1(V1)
nm=sqrt(sum( V1 .* conj( V1 )));
idx=(nm>1e-6);
V1(:,idx)=V1(:,idx)*diag(1./nm(idx));
V1= V1*diag(1./sign(V1(1,:)));
[~,idx] = sort(V1(1,:));
V1 = V1(:,idx);