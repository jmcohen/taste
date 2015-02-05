% This is an example of how to use NMF to find factor-specific genes and do gene set analysis.
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
% Feb. 26, 2012
%%%%

clear

% load data
% suppose the current folder is the one containing the NMF toolbox
load('.\data\ALLAML.mat');
classes=classes012;
clear('classes012');
k=3;

% both basis vectors and coefficient vectors are sparse and non-negative 
Dn=normc(D);
% set random number generator so as to same result can be produced
s = RandStream('swb2712','Seed',10);
RandStream.setDefaultStream(s);

optionsnmf2.alpha2=0.01;
optionsnmf2.alpha1=0.01;
optionsnmf2.lambda2=0.0;
optionsnmf2.lambda1=0.01;
optionsnmf2.t1=true;
optionsnmf2.t2=true;
% [A,Y]=sparsenmf2rule(Dn,k,optionsnmf2);
% [A,Y]=sparsenmf2nnqp(Dn,k,optionsnmf2);
[A,Y]=vsmf(Dn,k,optionsnmf2);

% gene ranking by gene score
optionffnmf.isBasis=true;
optionffnmf.propertyName='median';
optionffnmf.propertyValue=1;
mask1=featureFilterNMF(A,[],optionffnmf);

% for each basis vector, find the factor-specific genes
for i=1:k
    basisVec=A(:,i);
    maxVal=max(basisVec);
    maski=(basisVec>1e-3*maxVal); % nonzero logical indicators
    genei=gene(mask1&maski); % genes for the i-th factor
    fprintf('The number of genes in the %d-th factor is %d\n.',i,numel(genei));
    fileName=['ALLAMLFactor',num2str(i)];
    writeDir='.\data';
    writeGeneList(genei,fileName,writeDir); % write into txt file
end

% Now, in the data subfolder, you should see ALLAMLFactor1.txt,
% ALLAMLFactor2.txt, and ALLAMLFactor3.txt
% Login to the Onco-Express (http://vortex.cs.wayne.edu/ontoexpress)
% Input ALLAMLFactor1(2)(3).txt, and set all the 5000 genes in ALLAML as reference (in the ALLAMLProbeList.txt file).
% Then you can see the enrichment of gene ontology.






