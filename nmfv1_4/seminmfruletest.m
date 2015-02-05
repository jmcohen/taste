function [Y,numIter,tElapsed]=seminmfruletest(X,outTrain)
% Project test samples into the feature space, this function is called by
% function featureExtractionTest.
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
% May 01, 2011
%%%%

tStart=tic;
optionDefault.iter=1000;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;

A=outTrain.factors{1};
if isfield(outTrain,'option')
    option=outTrain.option;
else
    option=[];
end
option=mergeOption(option,optionDefault);

Y=A\X;
Y(Y<0)=0;
Apn=A'*X;
Ap=(abs(Apn)+Apn)./2;
An=(abs(Apn)-Apn)./2;
Bpn=A'*A;
Bp=(abs(Bpn)+Bpn)./2;
Bn=(abs(Bpn)-Bpn)./2;
XfitPrevious=Inf;
for i=1:option.iter
    Y=Y.*sqrt((Ap + Bn*Y)./(An + Bp*Y));
    if mod(i,10)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=matrixNorm(X-XfitThis);
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            disp(['Mutiple update rules based on SemiNMF successes!, # of iterations is ',num2str(i),'. The final residual is ',num2str(curRes)]);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
