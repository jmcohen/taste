% This is an example of how to use nmf to dicover biological process and how to draw
% the heatmap afterthat.
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
load('.\data\GSE3431.mat','dataUnikORF','unikORF');
dataUnikORF=normr(dataUnikORF); % normalize to have the effort of alignment
t=(1:36)'; % time points
optNNQP.lambda=2^-1;
option.algorithm='nmfnnls';
option.reorder=true;
k=3; % number of clusters
[indCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=NMFCluster(dataUnikORF',k,option);
hNMF=plot(t,Aout,'LineWidth',2);
set(gca,'XLim',[1,36]);
xlabel('Time Point');
ylabel('Intensity');
Legend(gca,{'cluster 1','cluster 2','cluster 3'});
print(gcf,'-depsc','-r200','bioPrecessJuly28.eps');
NMFHeatMap(Xout,Aout,Yout);

% [A,Y,numIter,tElapsed,finalResidual]=sparseNMFNNQP(dataUnikORF',3);
% plot(t,A);
% NMFHeatMap(dataUnikORF',A,Y);

