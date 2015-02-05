% This is an example of how to use NMF as clustering method
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
% Aug. 10, 2012
%%%%

clear

% load data
% suppose the current folder is the one containing the NMF toolbox
load('.\data\ALLAML.mat','classes012','D');
classes=classes012;
clear('classes012');
k=3;

% Directly call a NMF algorithm to obtain the factor matrices.
[A,Y]=nmfnnls(D,k);
% Then clustering. 
indCluster10=NMFCluster(Y);

Dn=normc(D);
optionsnmf2.alpha2=1;
optionsnmf2.alpha1=0.01;
optionsnmf2.lambda2=0;
optionsnmf2.lambda1=0.1;
[A,Y]=sparsenmf2nnqp(Dn,k,optionsnmf2);
% Then clustering. 
indCluster11=NMFCluster(Y);

% Call the NMF interface. The default is nmfnnls algorithm.
[A,Y]=nmf(D,k);
% Then clustering.
indCluster2=NMFCluster(Y);

% Call the NMF interface. Specify an algorithm.
option.algorithm='nmfnnls';
[A,Y]=nmf(D,k,option);
% Then clustering.
indCluster3=NMFCluster(Y);

% Call the NMF interface. Specify an algorithm.
option.algorithm='sparsenmfNNQP';
option.optionnmf.lambda=2^0;
[A,Y]=nmf(D,k,option);
% Then clustering.
indCluster4=NMFCluster(Y);

% Directly call NMFCluster function. NMF optimization is done inside. The
% default algorithm is nmfnnls.
indCluster5=NMFCluster(D,k);

% % Directly call NMFCluster function. NMF optimization is done inside.
% Specify an algorithm.
option.algorithm='convexnmfrule';
indCluster6=NMFCluster(D,k,option);




