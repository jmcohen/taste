function voted_label=wvote(classk,scorek)
% vote by the majority rule
% classk: matrix, rows are samples, columns are committee members
% voted_label, column vector
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 26, 2011
% % example:
% classk=[0,0,0,1,2;2,1,0,1,1];
% scorek=[0.5,0.2,0.2,0.1,0;0.11,0.1,0.095,0.092,0.09];
% voted_label=wvote(classk,scorek)

[numSample,numCom]=size(classk);
voted_label=nan(numSample,1);
for i=1:numSample
    cl=unique(classk(i,:));
    if numel(cl)==1
        voted_label(i)=cl;
        continue;
    end
    numVote=zeros(size(cl));
    class=classk(i,:);
    score=scorek(i,:);
    for j=1:numel(cl)
        numVote(j)=sum(score(class==cl(j)));
    end
    [maxVal,maxInd]=max(numVote);
    voted_label(i)=cl(maxInd);
end
end