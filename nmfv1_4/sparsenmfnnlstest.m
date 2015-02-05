function [Y,numIter,tElapsed]=sparsenmfnnlstest(X,outTrain)
% map the test/unknown samples into the sparse-NMF feature space
% X: matrix, test/unknown set, each column is a sample, each row is a feature.
% outTrain: struct, related options in the training step.
% outTrain.factors: column vector of cell of length 2, contain the matrix factors obtained by NMF
% optTrain.facts: scalar, the number of new features.
% outTrain.optionTr: struct, the option input into the training step. The default is [].
% Y: matrix, the test samples in the feature space.
% numIter: scalar, number of iterations.
% tElapsed: scalar, the computing consumed.
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
% May 03, 2011
%%%%


tStart=tic;

optionDefault.eta=(max(max(X)))^2;
optionDefault.beta=0.1;
optionDefault.iter=200;
optionDefault.dis=true;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if isfield(outTrain,'optionTr')
    option=outTrain.optionTr;
else
    option=[];
end
option=mergeOption(option,optionDefault);

A=outTrain.factors{1};
k=outTrain.facts;

if option.eta==0 || option.beta==0
    tElapsed=toc(tStart);
    [Y,numIter,tElapsed]=nmfnnlstest(X,outTrain);
    return;
end

[r,c]=size(X); % c is # of samples, r is # of features
Ae=[A;sqrt(option.beta)*ones(1,k)];
X0=[X;zeros(1,c)];
Y=kfcnnls(Ae,X0);
numIter=1;
tElapsed=toc(tStart);
end


