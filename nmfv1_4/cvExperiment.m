function [meanMeanPerformsAllRun,stdSTDAllRun,meanconMatAllRun,meantElapsedsAllRun]=cvExperiment(D,classes,kfold,rerun,fsMethod,fsOption,feMethod,feOption,classMethods,options)
% Conduct experiment of k-fold cross-validation on a dataset
% D: matrix, each row corresponds to a feature, each column a sample or
% data point
% classes: column vector, the class labels of the data points, it must be
% in the range [0,C-1].
% kfold: scalar, k-fold
% rerun: scalar, how many times do you want to rerun k-fold CV
% fsMethod: string, the feature selection method
% fsOption: struct, the option for feature selection
% feMethod: string, the feature extraction method
% feOption: struct, the option for feature extraction
% classMethods: cell vector of strings
% options: cell vector of options of classification
% meanMeanPerformsAllRun: (#class+2)*#classificationMethod matrix, the mean numeric performance over many runs
% stdSTDAllRun: (#class+2)*#classificationMethod matrix, the std over many runs
% meanconMatAllRun: matrix of #class*#class*#classificationMethod, the mean confusion matrix over many runs
% meantElapsedsAllRun: vector of length #classificationMethod, the mean time eclapsed over many runs
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
% July 18, 2012
%%%%

numMtds=numel(classMethods);
numClasses=numel(unique(classes));
numberPerfMeasure=2+numClasses;

% AllRecord=[];
classPerformsARun=nan(kfold,numberPerfMeasure,numMtds);
% conMatARun=nan(kfold,numClasses,numClasses,numMtds);
tElapsedsARun=nan(kfold,numMtds);

meanPerformsAllRun=nan(rerun,numberPerfMeasure,numMtds);
stdPerformsAllRun=nan(rerun,numberPerfMeasure,numMtds);
conMatAllRun=nan(rerun,numClasses,numClasses,numMtds);
tElapsedsAllRun=nan(rerun,numMtds);

for r=1:rerun
    % Cross Validation
    msg=sprintf('Rerun %i >>>>>>>>>>>>>>>>>>>>>>>>>',r);
    disp(msg);
    % reset the random number generator
    s = RandStream('swb2712','Seed',r);
    RandStream.setDefaultStream(s);
    ind=crossvalind('Kfold',classes,kfold);
    testClassPredictedsARun=[];
    testClassARun=[];
    for i=1:kfold
        indTest=(ind==i);
        if size(D,3)>1 % 3D data
            trainSet=D(:,:,~indTest);
            testSet=D(:,:,indTest);
        else % 2D data
            trainSet=D(:,~indTest);
            testSet=D(:,indTest);
        end
        
        trainClass=classes(~indTest);
        testClass=classes(indTest);
        testClassARun=[testClassARun;testClass];
        % feature extraction
        if ~strcmp(feMethod,'none') && feOption.searchKernelParam
            feOption.gammaRange=searchkernelRange(trainSet,trainClass,feOption);
            [gammaOptimal,KOptimal,accMax]=gridSearchGammaK(trainSet,trainClass,feMethod,feOption);
            feOption.facts=KOptimal;
            feOption.option.param=gammaOptimal;
        end
        [trainExtr,outTrain]=featureExtractionTrain(trainSet,testSet,trainClass,feMethod,feOption);
        [testExtr,outTest]=featureExtrationTest(trainSet,testSet,outTrain);
        trainSet=trainExtr;
        testSet=testExtr;
        
        % feature selection
        [featIndSelected,otherOutput]=featSel(trainSet,trainClass,fsMethod,fsOption);
        trainSet=trainSet(featIndSelected,:);
        testSet=testSet(featIndSelected,:);
        
        % classification
        [testClassPredicteds,classPerforms,conMats,tElapseds,OtherOutputs]=multiClassifiers(trainSet,trainClass,testSet,testClass,classMethods,options);
        testClassPredictedsARun=[testClassPredictedsARun;testClassPredicteds];
%         classPerformsARun(i,:,:)=classPerforms';%reshape(testClassPredicteds,1,size(testClassPredicteds,));
        tElapsedsARun(i,:)=tElapseds;
%         conMatARun(i,:,:,:)=conMats;
        
    end
    
    % performance of cross-validation
    for  m=1:numMtds
        [meanPerformsAllRun(r,:,m),conMatAllRun(r,:,:,m)]=perform(testClassPredictedsARun(:,m),testClassARun,numClasses);
    end
    
    tElapsedsAllRun(r,:)=sum(tElapsedsARun,1);

end

meanMeanPerformsAllRun=squeeze(nanmean(meanPerformsAllRun,1))';
stdSTDAllRun=squeeze(nanstd(meanPerformsAllRun,1))';
meanconMatAllRun=squeeze(mean(conMatAllRun,1));

meantElapsedsAllRun=mean(tElapsedsAllRun,1);
end
