function Pk=EQP(H,C,X,W,lambda)
% solve the equality constrained QP
% Yifeng Li, Feb. 21, 2012

option.tol=0;%2^(-32);
[nV,nP]=size(X);
% code=2.^(2*nV-1:-1:0) * W;
code=genCode(W);
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
   either=xor(Wi(1:nV),Wi(nV+1:end)); % either or but not both are constrained
   sign=-ones(nV,1);
   sign(either&Wi(1:nV))=1;
   sign=sign(either);
   numSameWorkSet=numel(indNum); % how many problem sharing this working set
   sign=repmat(sign,1,numSameWorkSet);
   Xie=X(either,indNum);
   Cie=C(either,indNum);
   Hie=H(either,either);
   Pkieither=(Hie+option.tol.*eye(size(Hie)))\(-(Hie*Xie+Cie+lambda.*sign));
   eitherNum=find(either); % numerical index of either
   Pk(eitherNum,indNum)=Pkieither;
   Pk(nV+eitherNum,indNum)=sign.*Pkieither;
end
end
