function [k,rho,kSuggested]=chooseBestk(X,option)
% Model selection. Search the best number of clusters based on dispersion Coefficients from k=1 to sqrt(#column).
% X: matrix, as input of a NMF, each column is a data point
% option: struct,
% option.rerun: scalar, the time of reruning NMF for a k, the default is 3;
% option.numClasses: scalar, number of the real classes, the default is [];
% option.threshold: if the dispersionCoefficient difference 
%     between the best k searched and the number of real classes <= threshold,
%     then, set the later as the suggested k.
% k: scalar, the best k.
% rho: column vector, the dispersion coefficients.
% kSuggested: scalar, the suggested k.
% References:
%  [1]\bibitem{NMFMetaBrunet2004}
%     J.P. Brunet, P. Tamayo, T.R. Golub, and J.P. Mesirov,
%     ``Metagenes and molecular pattern discovery using matrix factorization,''
%     {\it PNAS},
%     vol. 101, no. 12, pp. 4164-4169, 2004.
%  [2]\bibitem{NMF_Sparse_Kim2007}
%     H. Kim and H. Park,
%     ``Sparse non-negatice matrix factorization via alternating non-negative-constrained least squares for microarray data analysis,''
%     {\it Bioinformatics},
%     vol. 23, no. 12, pp. 1495-1502, 2007.
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
% May 26, 2011
%%%%


optionDefault.rerun=30;
optionDefault.numClasses=[];
optionDefault.threshold=0.02;

if nargin<2
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

numSample=size(X,2);
rho=zeros(round(sqrt(numSample)),1);
rho(1)=0;
negative=any(any(X<0));
for k=2:round(sqrt(numSample))
    C=zeros(numSample,numSample);
    for r=1:option.rerun
        if negative
            [A,Y]=seminmfnnls(X,k);
        else
            [A,Y]=nmfnnls(X,k);
        end
        ind=getClusters(Y);
        C=C+getRelationMatrix(ind);
    end
    C=C./option.rerun;
    rho(k)=dispersionCoefficient(C);
end
[val,k]=max(rho);
% if there are multiple identical maximal vals, return index of the last one
k=find(rho==val);
k=k(end);
% if the number of classes exists, if its rho is close to the optimal rho, then suggest to use the number of classes  
kSuggested=k;
if isfield(option,'numClasses')&& ~isempty(option.numClasses)
    if abs(rho(k)-rho(option.numClasses))<=option.threshold
        kSuggested=option.numClasses;
    end
end
end
