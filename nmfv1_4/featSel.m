function [featIndSelected,otherOutput]=featSel(trainSet,trainClass,fsMethod,option)
% feature selection
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% fsMethod: string, by now it can be 'none','ttest', and 'nmf'. More
% methods will be added in later version.
% option: struct, the option for a feature selection method, see the
% specific feature selection method for details.
% featIndSelected: column vector, the numeric indices
% otherOutput: other output, by now is it [].
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
% Dec. 13, 2011
%%%%

if nargin<4
   option=[]; 
end
[numFeat,numSampleTrain]=size(trainSet);
switch fsMethod
    case 'none'
        featIndSelected=(1:numFeat)';
        otherOutput=[];
        return;
    case 'ttest' % t-test
        optionDefault.d=20;
        option=mergeOption(option,optionDefault);
        [featIndSelected, Z] = rankfeatures(trainSet, trainClass, 'Criterion', 'ttest','NumberOfIndices',option.d); %'entropy', 'bhattacharyya','roc','wilcoxon'
        otherOutput=[];
    case 'nmf'
        [mask,featNames,scores,A]=featureFilterNMF(trainSet,[],option);
        featIndSelected=(1:numFeat)';
        featIndSelected=featIndSelected(mask);
        otherOutput=[];
    case 'vsmf'
        [mask,featNames,scores,A]=featSelNMF(trainSet,[],option);
        featIndSelected=(1:numFeat)';
        featIndSelected=featIndSelected(mask);
        otherOutput=[];
    otherwise 
        error('Please select a correct feature selection method.');
end

end