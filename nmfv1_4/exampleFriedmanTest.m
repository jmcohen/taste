% This is an example of how to conduct Friedman test and plot CD diagram
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
% Nov. 29, 2012
%%%%

clear
classifiers={'NNLS-MAX';'NNLS-NS';'SRC';'MSRC';'LRC';'1-NN';'SVM';'ELM'};
Accs=zeros(13,8); % dataset by classifier
Accs=[0.9176    0.9696    0.9707    0.9714    0.9673    0.9539    0.9678    0.7729; % dataset 1
      0.7984    0.8566    0.8609    0.7281    0.8476    0.7831    0.8922    0.7356;
      0.7519    0.8590    0.8526    0.8901    0.7536    0.8012    0.8134    0.7180;
      0.9327    0.9612    0.9598    0.9504    0.9406    0.9126    0.9709    0.8182;
      0.8484    0.9253    0.9228    0.7150    0.8699    0.7890    0.9736    0.8776;
      0.7169    0.7801    0.7788    0.8084    0.7526    0.7447    0.8103    0.6982;
      0.8419    0.9214    0.9294    0.9317    0.9272    0.9228    0.9528    0.8061;
      0.9296    0.9764    0.9790    0.9197    0.9759    0.9403    0.9819    0.8568;
      0.9625    0.9854    0.9856    0.8672    0.9842    0.9686    0.9817    0.9785;
      0.8722    0.9175    0.9122    0.7675    0.9622    0.8312    0.9218    0.8990;
      0.8843    0.9452    0.9537    0.7900    0.9673    0.8626    0.9317    0.9004;
      0.9027    0.9331    0.9253    0.7543    0.8090    0.8710    0.9286    0.8913;
      0.7613    0.8323    0.8306    0.7050    0.8745    0.7428    0.8595    0.8083]; % dataset 13
error=1-Accs;
alpha=0.05;
% do Friedman test with post-hoc Nemenyi test
[F,pvalF,acceptF,rankCl,CD]=FriedmanTest(error,alpha);
% plot Nemenyi test and save in current folder
saveFigName='NemenyiTestPlotHigh.eps';
plotNemenyiTest(rankCl,CD,classifiers,saveFigName);