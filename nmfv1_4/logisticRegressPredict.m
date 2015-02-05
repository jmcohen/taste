function testClassPredicted=logisticRegressPredict(model,testSet)
% predict the class labels of unknown samples using the model learned by
% logisticRegressTrain
% for example
% load('C:\YifengLi\Reseach Program\lisa\Hu2006UniqueUnigene.mat','DOrdered','shortenClassesOrdered','classesOrdered','annotUnikUG');
% dataStr='Hu2006';
% % remove Claudin
% rmInd=strcmp(shortenClassesOrdered,'Claudin');
% shortenClassesOrdered=shortenClassesOrdered(~rmInd);
% classesOrdered=classesOrdered(~rmInd);
% DOrdered=DOrdered(:,~rmInd);
% D=DOrdered;
% classes=classesOrdered;
% D=knnimpute(D);
% D=D(1:100,:);
% classes=changeClassLabels01(classes);
% ind=crossvalind('Kfold',classes,3);
% indTest=(ind==1);
% trainSet=D(:,~indTest);
% testSet=D(:,indTest);
% trainClass=classes(~indTest);
% testClass=classes(indTest);
% option=[];
% model=logisticRegressTrain(trainSet,trainClass,option);
% testClassPredicted=logisticRegressPredict(model,testSet);
% [perf,conMat]=perform(testClassPredicted,testClass,5)
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
% July 04, 2012
%%%%

% numTe=size(testSet,2);
% ssize=ones(numTe,1);
Y = mnrval(model.B,testSet');
testClassPredicted=getLabel(Y);
testClassPredicted=testClassPredicted-1;
end