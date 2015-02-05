function [Y,numIter,tElapsed]=convexnmfruletest(X,outTrain)
% map the test/unknown samples into the convex-NMF feature space, this
% function is called by function featureExtractionTest
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

optionDefault.iter=1000;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if isfield(outTrain,'optionTr')
    option=outTrain.optionTr;
else
    option=[];
end
option=mergeOption(option,optionDefault);

Xtrain=outTrain.factors{1};
A=outTrain.factors{2};
Y=outTrain.factors{3};
%k=outTrain.facts;

outTrain.factors=[];
outTrain.factors{1}=Xtrain*A;
outTrain.factors{2}=Y;
% call nmfnnlstest
[Y,numIter]=nmfnnlstest(X,outTrain);


% Ak=Xtrain'*Xtrain;
% Ap=(abs(Ak)+Ak)./2;
% An=(abs(Ak)-Ak)./2;
% [r,c]=size(X); % c is # of samples, r is # of features
% inx=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
% H=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
% G=H+0.2;
% W=A;
% XfitPrevious=Inf;
% ApW=Ap*W;
% AnW=An*W;
% for i=1:option.iter    
%     GWt=G*W';
%     G=G.*sqrt((ApW + GWt*AnW)./(AnW + GWt*ApW));
% %     G(G<eps)=0;
%     G=max(G,eps);
%     Y=G';
%     if mod(i,10)==0 || i==option.iter
%         if option.dis
%             disp(['Iterating >>>>>> ', num2str(i),'th']);
%         end
%         XfitThis=XTrain*A*Y;
%         fitRes=norm(XfitPrevious-XfitThis,'fro');
%         XfitPrevious=XfitThis;
%         curRes=norm(X-XfitThis,'fro');
%         if option.tof>=fitRes || option.residual>=curRes || i==option.iter
%             s=sprintf('Mutiple update rules based ConvexNMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
%             disp(s);
%             numIter=i;
%             finalResidual=curRes;
%             break;
%         end
%     end
% end
tElapsed=toc(tStart);
end
