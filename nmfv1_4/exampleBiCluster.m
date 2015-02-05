% This is an example of how to use the "biCluster" finction and how to draw
% the heatmap after biclustering.
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
k=3; % number of cluster

option.propertyName='threshold';
option.propertyValue=0.9;
option.algorithm='nmfnnls';
[ACluster,YCluster,indACluster,indYCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=biCluster(D,k,option);

option.colormap=redgreencmap; % redgreencmap
option.standardize=true;
NMFBicHeatMap(Xout,Aout,Yout,indACluster,indYCluster,k,option);
