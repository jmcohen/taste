function testClassPredicted=knnrule(Y,trainClass,k)
% knn rule for sparse coding based classification
% Y: matrix of size K times n, each column is a the sparse coding 
% trainClass: n-length vector, the class labels
% k: the number of largest coefficients to make decision
% Yifeng Li
% Feb. 14, 2012

%numTr=numel(trainClass);
[numTr,numTe]=size(Y);
if nargin<3
   k=numTr; 
end
if k>numTr
   k=numTr; 
end
classk=zeros(numTe,k);
scorek=zeros(numTe,k);
for i=1:numTe
    [sortedCoeff,ind] = getBestScores(Y(:,i),k);
    classk(i,:)=trainClass(ind);
    scorek(i,:)=sortedCoeff;
end
testClassPredicted=wvote(classk,scorek);
end



        
        
        
