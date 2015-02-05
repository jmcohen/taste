function [testClassPredicted,residuals]=subspace(Y,testSet,trainSet,trainClass)
% the nearest subspace (NS) rule
% Y, matrix, the spase matrix, each column is a coefficient vector for a
% testing sample.
% testSet: matrix, the test samples
% trainSet: matrix, each column is a training sample
% trainClass: column vector, the class labels for training samples
% testClassPredicted: column vectors, the predicted class labels of
% testing samples.
% residuals: matrix of size #testsamples by #classes, the residuals.
% residuals[i,j] is the residual of testing sample i in class j.
% Yifeng Li


utrCl=unique(trainClass);
numCl=numel(utrCl);
numTest=size(Y,2);
testClassPredicted=zeros(numTest,1);
residuals=zeros(numTest,numCl);
ifMissVal=any(any(isnan([trainSet,testSet])));
% no missing values
if ~ifMissVal
    for s=1:numTest
        residual=zeros(numCl,1);
        for i=1:numCl
            ind=(trainClass==utrCl(i));
            Yis=Y(ind,s);
            trainSeti=trainSet(:,ind);
            residual(i)=matrixNorm(testSet(:,s)-trainSeti*Yis);
            residuals(s,i)=residual(i);
        end
        [val,resInd]=min(residual);
        testClassPredicted(s)=utrCl(resInd);
    end
end
% missing values
if ifMissVal
    % precompute inner product of each class
    innerProductClasses=cell(numCl,1);
    for i=1:numCl
        ind=(trainClass==utrCl(i));
        trainSeti=trainSet(:,ind);
        innerProductClasses{i}=innerProduct(trainSeti,trainSeti);
    end
    
    for s=1:numTest
        residual=zeros(numCl,1);
        for i=1:numCl
            ind=(trainClass==utrCl(i));
            Yis=Y(ind,s);
            trainSeti=trainSet(:,ind);
            residual(i)=innerProduct(testSet(:,s),testSet(:,s))-2*(innerProduct(testSet(:,s),trainSeti)*Yis) + Yis'*innerProductClasses{i}*Yis; 
        end
        [val,resInd]=min(residual);
        testClassPredicted(s)=utrCl(resInd);
    end
end    
end