function dist=dif(U,Utrue,measure)
dist=100;
if nargin==2
    measure='fro';
end
U=sort1(U); 
[~,match_idx]=max(abs(Utrue'*U),[],1);
if length(unique(match_idx))==size(Utrue,2)
    dist=norm(U(:,match_idx)-Utrue,measure);
end