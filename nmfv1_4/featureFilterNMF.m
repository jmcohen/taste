function [mask,featNames,scores,A]=featureFilterNMF(D,featNames,opt)
% feature filtering using NMF
% Usage:
% [mask,featNames,A,scores]=featureFilterNMF(D)
% [mask,featNames,A,scores]=featureFilterNMF(D,featNames)
% [mask,featNames,A,scores]=featureFilterNMF(D,featNames,opt)
% [mask,featNames,A,scores]=featureFilterNMF(D,[],opt)
% D: matrix, 
%     if opt.isBasis=false, (default) D is nonnegative training data. The rows correspond features, columns correspond sample;
%     if opt.isBasis=true, D is basis matrix by a NMF outside this function.
% featNames, column vector of string cell, the names of features
% opt: struct:
% opt.facts: the number of columns in the basis matrix
% option.isBasis: boolen scalar, true if D is basis matrix, false if D is training data
% opt.propertyName: string, the way to set the threshold. It could be
%     'threshold': scalar from [0,1], if score of a feature is less than opt.PropertyValue, then remove this feature
%                  The default opt.PropertyValue is 0.5.
%     'percentile': scalar from [0,1], keep opt.PropertyValue of features and remove 1-opt.PropertyValue features
%                  The default opt.PropertyValue is 0.2.
%     'number': scalar from [1,#features], keep opt.PropertyValue of features
%                  The default opt.PropertyValue is 0.2*#features.
%     'median': scalar from [0,1], set the threshold median + opt.PropertyValue * mad of the scores
%                  The default opt.PropertyValue is 0.
% opt.PropertyValue: scalar, see above
% opt.facts: scalar, number of columns in the basis matrix in NMF
% opt.optionn: sturct, option for the specified nmf function, type "help nmf" in the command line for more information
% mask: column of boolen vector of length size(D,1), 
%     mask[i]=true if the ith feature is selected, otherwise false.
% featNames: column vector of string cell, the names of the selected features
% scores: column vector, the scores of the selected features
% A: matrix, the basis matrix with the selected features
% Reference:
%  [1]\bibitem{NMF_Sparse_Kim2007}
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
% May 20, 2011
%%%%

if nargin<3
   opt=[]; 
end
if nargin<2
   featNames=[]; 
end

optionDefault.isBasis=false;
optionDefault.propertyName='threshold';
optionDefault.propertyValue=0.5;
optionDefault.facts=2;
optionDefault.optionn=[];
opt=mergeOption(opt,optionDefault);

if ~opt.isBasis
     [A,Y]=nmf(D,opt.facts,opt.optionn);
else
    A=D;
end
clear('D');

A=A(:,(sum(abs(A))>eps)); % remove null columns
% A=normc(A);
A=abs(A); % absolute values
[numR,numC]=size(A);
% aviod divided by 0; p=F./(sum(F,2)*[1,1]);
p=zeros(numR,numC);
sumA=sum(A,2);
for i=1:numR
    if sumA(i)~=0
        p(i,:)=A(i,:)./sumA(i);
    end
end
% aviod 0*log0=nan. let 0*log0=0
plog2p=zeros(numR,numC);
for i=1:numR
    for j=1:numC
        if p(i,j)~=0
            plog2p(i,j)=p(i,j)*log2(p(i,j));
        end
    end
end
featScores=1+(1./log2(numC)).*sum(plog2p,2);

switch opt.propertyName
    case 'percentile'
        if isempty(opt.propertyValue)
           numGeneSel=round(0.2*numR);
        else
           numGeneSel=round(opt.propertyValue*numR);%size(D,1)-round(size(D,1)*(PropertyValue/100));
        end
        if numGeneSel<1
               numGeneSel=1;
        end
        bestK=numGeneSel;
        [Kscores,nfindex]=getBestScores(featScores,bestK);
        mask=false(numR,1);
        mask(nfindex)=true;
    case 'number'
        if isempty(opt.propertyValue)
           numGeneSel=0.2*numR;
        else
           numGeneSel=opt.propertyValue;
        end
        if numGeneSel<1
               numGeneSel=1;
        end
        bestK=numGeneSel;
        [Kscores,nfindex]=getBestScores(featScores,bestK);
        mask=false(numR,1);
        mask(nfindex)=true;
    case 'threshold'
        if isempty(opt.propertyValue)
           threshold=0.5;
        else
           threshold=opt.propertyValue;
        end
         mask=(featScores>=threshold);
    case 'median'
        if isempty(opt.propertyValue)
           times=0;
        else
           times=opt.propertyValue;
        end
        m=median(featScores);
        d=mad(featScores,1);
        mask=(featScores>=m+times*d);
end
scores=featScores(mask);
A=A(mask,:);
if ~isempty(featNames)
    featNames=featNames(mask);
end
end