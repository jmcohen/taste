function indLog=leaveMOut(m,C)
% leave one out
% C can be scalar or vector
% C: if C is scalar, it is the number of samples; if C is a vector, it is the class labels
% Yifeng Li
% Oct 29, 2012
% example: 
% m=10;
% C=[0;0;0;0;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2];
% indLog=leaveMOut(m,C)

numS=numel(C);

if numS==1 % C is the total number of samples
    numS=C;
    indLog=false(C,1);
    % to be finish
else % numS is the number of samples
    indLog=false(numS,1);
    unikCl=unique(C);
    numUnikCl=numel(unikCl);
    numEachCl=zeros(numUnikCl,1);
    for c=1:numUnikCl
        numEachCl(c)=sum(C==unikCl(c));
    end
    % sort in incremental order
    [numEachCl,indx]=sort(numEachCl);
    unikCl=unikCl(indx);
    % number of samples to be taken from each class
    ratio=m/numS;
    eachClTaken=zeros(numUnikCl,1);
    for c=1:numUnikCl-1
        eachClTaken(c)=round(ratio*numEachCl(c));
        if eachClTaken(c)==0
            eachClTaken(c)=1;
        end
    end
    eachClTaken(numUnikCl)=m-sum(eachClTaken);
    if numEachCl<=0
        error('The last class has zeros samples');
    end
end

% take samples from each class
for c=1:numUnikCl
    indNum=(1:numS)';
    indNumCl=indNum(C==unikCl(c));
    perm=randperm(numEachCl(c));
    indNumClRank=indNumCl(perm);
    indNumClRankTaken=indNumClRank(1:eachClTaken(c));
    indLog(indNumClRankTaken)=true;
end
end
