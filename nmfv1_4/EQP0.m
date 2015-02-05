function Pk=EQP0(H,C,X,W,A,B)
% solve the equality constrained QP
%%%%
% Copyright (C) <2012>  <Yifeng Li>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% Feb. 21, 2012
%%%%

option.tol=0;%2^(-32);
[nV,nP]=size(X);
% code=2.^(2*nV-1:-1:0) * W;
code=geneCode(W);
% [codeSorted,csInd]=sort(code);
codeUnik=unique(code);
numUnik=numel(codeUnik);
Pk=zeros(2*nV,nP);
for i=1:numUnik
   indLog=(code==codeUnik(i)); % index of the same working set
   indNum=find(indLog);
   Wi=W(:,indNum(1)); % this unique working set
   if any(~(Wi(1:nV)|Wi(nV+1:end)))
      error('there exist unbounded problem'); 
   end
   NZero=(Wi(1:nV)==false); % constraints of x_i~=0
   Xi=X(NZero,indNum);
   Ci=C(NZero,indNum);
   Hi=H(NZero,NZero);
   if Wi(end) % the last is active
        Ai=A(NZero);
        Bi=B(indNum);
        gi=Gi*Xi + Ci;
        hi=Ai*Xi-Bi;
        nPkLamda=([Hi,Ai';Ai,0]+option.tol.*eye(size(Hie)))\[gi;hi];
        Pki=-nPkLamda(1:nV);
   else
       gi=Gi*Xi + Ci;
       Pki=(Gi+option.tol.*eye(size(Hie)))\gi;
   end
   Pk(NZero,indNum)=Pki;
end
end



















