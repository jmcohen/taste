function P=innerProduct(A,B,option)
% compute the inner product of two matrices which containing missing values
% A: matrix
% B: matrix
% P, matrix, the inner product of A and B. That is A'*B;
% Yifeng Li

[ra,ca]=size(A);
[rb,cb]=size(B);
if ra~=rb
    P=[];
    error('A and B should have the same number of rows'); 
end
P=zeros(ca,cb);
ifA=~isnan(A);
ifB=~isnan(B);
for i=1:ca
   for j=1:cb
       ind=(ifA(:,i) & ifB(:,j));
       ai=A(ind,i);
       bj=B(ind,j);
       P(i,j)=(ai'*bj)/(matrixNorm(ai)*matrixNorm(bj));
   end
end
end