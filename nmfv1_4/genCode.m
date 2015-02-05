function codes=genCode(W)
% generate codes for columns of logical matrix W
% W: logical, matrix of m x n
% code: column vector of n x 1
% Yifeng Li
% Mar. 18, 2013
% example:
% W=[true,false,false,true,true,true,;
%    false,false,true,false,true,true];
% codes=genCode(W)

[m,n]=size(W);
codes=zeros(n,1);
for i=1:n
    if codes(i)==0
        codes(i)=i;
    end
   for j=i+1:n
      if codes(j)==0 && sum(W(:,i)==W(:,j))==m
         codes(j)=i; 
      end
   end
end
end