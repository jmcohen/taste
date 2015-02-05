function plotNemenyiTest(rankCl,CD,classifiers,saveFigName)
% plot the graph of the result of Nemenyi test, this function is usually
% used after the FriedmanTest function
% rankCl: column vector, the ranks of all classifiers
% CD: scalar, the critical difference
% saveFigName: string, the file name to be saved in the current folder
% Yifeng Li
% October 24, 2012
% example: 
% rankCl=[1.20; 2.15; 2.50; 3.7; 3.90];
% CD=1;
% classifiers={'Classifier1';'Classifier2';'Classifier3';'Classifier4';'Classifier5'};
% plotNemenyiTest(rankCl,CD)

if nargin<4
   saveFigName='NemenyiTest.eps';
end

% sort the ranks
[rankCl,idx]=sort(rankCl);
classifiers=classifiers(idx);

numCl=numel(rankCl);
figure('OuterPosition',[40,40,1000,600],...
'MenuBar','figure',...
'ToolBar','figure',...
'PaperPositionMode','auto');
for i=1:numCl;
    plot(gca,[rankCl(i),rankCl(i)],[0,i],'k','LineWidth',1.5);
    xlim([1,numCl]);
    ylim([0,numCl+1.5])
    x=[rankCl(i) rankCl(i)];
    y=[i+0.5 i];
    [xf,yf]=ds2nfu(x,y);
    txtar=annotation('textarrow',xf,yf,...
                   'String',classifiers{i},'FontSize',14);
    hold on;
end
plot(gca,[1,1+CD],[numCl+1,numCl+1],'k','LineWidth',1.5);
x=[1+CD/2 1+CD/2];
    y=[numCl+0.5 numCl+1];
    [xf,yf]=ds2nfu(x,y);
txtar=annotation('textarrow',xf,yf,...
                   'String','CD','FontSize',14);
% connect groups
for i=1:numCl-1
   for j=i+1:numCl
       if rankCl(j)-rankCl(i)>=0 && rankCl(j)-rankCl(i) <= CD
          plot(gca,[rankCl(i)-0.05,rankCl(j)+0.05],[i-0.5,i-0.5],'k','LineWidth',2.5);
          hold on;
       end
   end
end
               
%set(gca,'XDir','reverse');
xlabel('Rank','FontSize',14);
set(gca,'FontSize',14);
hold off;
print(gcf,'-depsc2','-r300',saveFigName);
end
