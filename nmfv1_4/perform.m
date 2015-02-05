function [perf,conMat]=perform(testClassPredicted,testClass,c)
% Analyze the performance of classification
% For binary-class problem, the class labels should be 0 and 1.
% For m-class problem, the class labels should be 0,1,2,3,...,m-1.
% Usage:
% perf=perform(testClassPredicted,testClass,c)
% testClassPredicted: vector, predicted class labels of testing samples.
% testClass: vector, the real class labels of testing.
% c: the number of classes.
% perf: row vector,
%      if c==2: perf=[Spec.,Sen., Acc., and BAcc]
%      if c>2: perf=[Acc1,Acc2,...,AccC,Acc,Bacc].
% for example,
% testClass=randi([0,2],100,1);
% testClassPredicted=randi([0,2],100,1);
% [perf,conMat]=perform(testClassPredicted,testClass,3)
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
% Oct. 24, 2009
%%%%

perf=zeros(1,c+2);
tfnum=isnumeric(testClass);
if ~tfnum
    error('The elements must be numeric: 0,1,...');
end
n=numel(testClass); % number of test samples

conMat=zeros(c,c); % confusion matrix

% fill out confusion matrix
for i=1:n
    if isinf(testClassPredicted(i))||isnan(testClassPredicted(i))
        continue;
    end
    conMat(testClass(i)+1,testClassPredicted(i)+1)=conMat(testClass(i)+1,testClassPredicted(i)+1) + 1;
end

% numerical measures
perf(1:c)=diag(conMat);
perf(1,c+1)=sum(perf(1:c)); % acc
% get acc0,acc1,...
for ci=1:c
    numci=sum(testClass==ci-1);
    perf(1,ci)=perf(1,ci)/numci;
end
perf(1,c+1)=perf(1,c+1)/n; %acc
perf(1,c+2)=nanmean(perf(1:c)); % geometricMean(perf(1:c)) % nanmean(perf(1:c)); % bacc/arithmetric mean
end

% the following is the original code

% if c<=4
%     perf=zeros(1,6);
% else
%     perf=zeros(1,c+2);
% end
% tfnum=isnumeric(testClass);
% if ~tfnum
%     error('The elements must be numeric: 0,1,...');
% end
% n=numel(testClass);
%
% conMat=zeros(c,c); % confusion matrix
%
% if c==2 % binary classes
% TP=0;
% TN=0;
% FP=0;
% FN=0;
% for i=1:n
%     if ~isnan(testClassPredicted(i))
%         if testClassPredicted(i)==1
%             if testClass(i)==1 % positive
%                 TP=TP+1;
%             else
%                 FP=FP+1;
%             end
%         else
%             if testClass(i)==0 % negtive
%                 TN=TN+1;
%             else
%                 FN=FN+1;
%             end
%
%         end
%     end
% end
% conMat(1,1)=TN;
% conMat(1,2)=FP;
% conMat(2,1)=FN;
% conMat(2,2)=TP;
% perf=zeros(1,6);
% perf(1,1)=TP/(TP+FP); %PPV
% perf(1,2)=TN/(TN+FN); %NPV
% perf(1,3)=TN/(TN+FP); %Spec.
% perf(1,4)=TP/(TP+FN); %Sen.
% perf(1,5)=(TP+TN)/n; %Accuracy
% perf(1,6)=sqrt(perf(1,3)*perf(1,4)); % Geometric mean %nanmean([perf(1,3);perf(1,4)]); % BACC
% end
%
% if c>=3 % multiple class
%     for i=1:n
%         if isinf(testClassPredicted(i))||isnan(testClassPredicted(i))
%            continue;
%         end
%         conMat(testClass(i)+1,testClassPredicted(i)+1)=conMat(testClass(i)+1,testClassPredicted(i)+1) + 1;
%     end
%     perf(1:c)=diag(conMat);
%     perf(1,c+1)=sum(perf(1:c)); % acc
%     % get acc0,acc1,...
%     for ci=1:c
%         numci=sum(testClass==ci-1);
%         perf(1,ci)=perf(1,ci)/numci;
%     end
%     perf(1,c+1)=perf(1,c+1)/n; %acc
%     perf(1,c+2)=geometricMean(perf(1:c)); % geometric mean % nanmean(perf(1:c)); % bacc
% end
