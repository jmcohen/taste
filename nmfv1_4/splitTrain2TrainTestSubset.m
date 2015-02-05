function [trainSetTrainSubset,trainClassTrainSubset,trainSetTestSubset,trainClassTestSubset]=splitTrain2TrainTestSubset(trainSet,trainClass,option)
% as the function name
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

if nargin<3
   option=[]; 
end
optionDefault.kfold=5;
option=mergeOption(option,optionDefault);
trInd=crossvalind('Kfold',trainClass,option.kfold);
trIndValid=(trInd==option.kfold);
trainSetTrainSubset=trainSet(:,~(trIndValid));
trainSetTestSubset=trainSet(:,trIndValid);
trainClassTrainSubset=trainClass(~(trIndValid));
trainClassTestSubset=trainClass(trIndValid);
end