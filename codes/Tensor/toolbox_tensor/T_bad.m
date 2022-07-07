function V0= T_bad(TE,r)
[n1,n2,n3] = size(TE);
V0{1} = sort1(randn(n1,r));
V0{2} = sort1(randn(n2,r));
V0{3}  =sort1(randn(n3,r));
V0{4}=rand(r,1);
end