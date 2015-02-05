% This is an example of how to use weighted NMF on data with missing values.
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

% simulate missing values
D(rand(size(D))<0.05)=NaN;
k=3; % number of cluster
% Directly call the weighted NMF algorithm to obtain the factor matrices.
[A,Y]=wnmfrule(D,k);
% Then clustering. 
indCluster1=NMFCluster(Y);

% Or, can the "nmf" function
option.algorithm='wnmfrule';
[A,Y]=nmf(D,k,option);
% Then clustering. 
indCluster2=NMFCluster(Y);

