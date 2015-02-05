function voted_label=vote(predicted_labels)
% vote by the majority rule
% predicted_labels: matrix, rows are samples, columns are committee members
% voted_label, column vector
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 26, 2011

[numSample,numCom]=size(predicted_labels);
voted_label=nan(numSample,1);
for i=1:numSample
    cl=unique(predicted_labels(i,:));
    if numel(cl)==1
        voted_label(i)=cl;
        continue;
    end
    numVote=zeros(size(cl));
    for j=1:numCom
        for c=1:numel(cl)
            if predicted_labels(i,j)==cl(c)
                numVote(c)=numVote(c)+1;
                break;
            end
        end
    end
    [maxVal,maxInd]=max(numVote);
    voted_label(i)=cl(maxInd);
end
end