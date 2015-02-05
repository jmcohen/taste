function [trainExtr,out,time]=featureExtractionTrain(trainSet,testSet,trainClass,feMethod,opt)
% Extract features from training set
% Usuage:
% [trainExtr,out]=featureExtractionTrain(trainSet,trainClass,feMethod)
% [trainExtr,out]=featureExtractionTrain(trainSet,trainClass,feMethod,opt)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% feMethod: string, feature extraction methods. It could be
%     'non': there is no feature extraction, return the input trainSet;
%     'nmf': the standard NMF;
%     'sparsenmf': sparse-NMF;
%     'orthnmf': orthogonal-NMF;
%     'seminmf': semi-NMF;
%     'convexnmf': convexc-NMF;
%     'knmf-dc': knmf-dc;
%     'knmf-cv': knmf-cv, kernel convex NMF;
%     'knmf-nnls': knmf-nnls, kernel semi-NMF using NNLS algorithm;
%     'knmf-ur': knmf-ur, kernel semi-NMF using update-rule algorithm;
%     'vsmf': versatile sparse matrix factorization 
%     'pca': PCA.
% opt: struct, the options to configure the feature extraction.
% opt.facts: scalar, the number of new features to extract.
% opt.option: options for specific algorithm. If feMethod is
%     'non': opt.option is invalid;
%     'nmf': type "help nmfnnls" for more information;
%     'sparsenmf': type "help sparsenmfnnls" for more information;
%     'orthnmf': type "help orthnmfrule" for more information;
%     'seminmf': type "help seminmfnnls" for more information;
%     'convexnmf': type "help convexnmfrule" for more information;
%     'knmf-dc': type "help kernelnmfdecom" for more information;
%     'knmf-cv': type "help kernelconvexnmf" for more information;
%     'knmf-nnls': type "help kernelseminmfnnls" for more information;
%     'knmf-ur': type "help kernelseminmfrule" for more information;
%     'pca': opt.option is invalid.
% trainExtr: matrix, the training data in the feature space.
% out: struct, include the feature extraction model information.
%     out.feMethod: the same as the input argument "feMethod";
%     out.facts: the same as opt.fact;
%     out.factors: column vector of cell, the factor matrices produced by specific algorithm;
%     out.optionTr: struct, = opt.option.
% Reference:
% [1]\bibitem{bibm10}
%     Y. Li and A. Ngom,
%     ``Non-negative matrix and tensor factorization based classification of clinical microarray gene expression data,''
%     {\it IEEE International Conference on Bioinformatics \& Biomedicine},
%     2010, pp.438-443.
% [2]\bibitem{cibcb2012}
%     Y. Li and A. Ngom,
%     ``A New Kernel Non-Negative Matrix Factorization and Its Application in Microarray Data Analysis,''
%     {\it CIBCB},
%     pp. 371-378, 2012.
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
% May 22, 2011
%%%%

if nargin<5
    opt=[];
end
optDefault.facts=numel(unique(trainClass));
optDefault.option=[];
opt=mergeOption(opt,optDefault);

tStart=tic;
switch feMethod
    case 'none'
        trainExtr=trainSet;
        out.feMethod=feMethod;
    case 'nmf'
        %         if ~isfield(opt,'facts')
        %            opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [A,trainExtr]=nmfnnls(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'sparsenmf'
        %         if ~isfield(opt,'facts')
        %            opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [A,trainExtr]=sparsenmfnnls(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'orthnmf'
        %         if ~isfield(opt,'facts')
        %            opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [A,S,trainExtr]=orthnmfrule(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={A;S;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-nnls'
        %         if ~isfield(opt,'facts')
        %             opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [AtA,trainExtr,~,~,~]=kernelseminmfnnls(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-l1nnls'
        %         if ~isfield(opt,'facts')
        %             opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [AtA,trainExtr,~,~,~]=kernelsparseseminmfnnls(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-l1nnqp'
        %         if ~isfield(opt,'facts')
        %             opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [AtA,trainExtr,~,~,~]=kernelSparseNMFNNQP(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-ur'
        [AtA,trainExtr,~,~,~]=kernelseminmfrule(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-dc'
        [A,trainExtr,~,~,~]=kernelnmfdecom(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'knmf-cv'
        [XtX,W,trainExtr,~,~,~]=kernelconvexnmf(trainSet,opt.facts,opt.option);
        AtA=W'*XtX*W;
        out.feMethod=feMethod;
        out.factors={AtA;W;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'ksr-nnls'
        opt.option.SRMethod='nnls';
        % dictionary learning
        trainTrain=computeKernelMatrix(trainSet,trainSet,opt.option);
        [AtA,trainExtr,~,~,~]=KSRDL(trainTrain,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'ksr-l1nnls'
        opt.option.SRMethod='l1nnls';
        % dictionary learning
        trainTrain=computeKernelMatrix(trainSet,trainSet,opt.option);
        [AtA,trainExtr,~,~,~]=KSRDL(trainTrain,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;trainExtr;trainSet};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'seminmf'
        %         if ~isfield(opt,'facts')
        %            opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [A,trainExtr]=seminmfnnls(trainSet,opt.facts,opt.option);%seminmfrule
        out.feMethod=feMethod;
        out.factors={A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'convexnmf'
        %         if ~isfield(opt,'facts')
        %            opt.facts=numel(unique(trainClass));
        %         end
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        [A,trainExtr]=convexnmfrule(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={trainSet;A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'transductive-nmf'
        numTr=size(trainSet,2);
%         numTe=size(testSet,2);
        [AtA,trainTestExtr,~,~,~]=kernelseminmfnnls([trainSet,testSet],opt.facts,opt.option);
        out.feMethod=feMethod;
        trainExtr=trainTestExtr(:,1:numTr);
        testExtr=trainTestExtr(:,numTr+1:end);
        out.factors={AtA;trainExtr;trainSet;testExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    case 'vsmf'
        [A,trainExtr,AtA]=vsmf(trainSet,opt.facts,opt.option);
        out.feMethod=feMethod;
        out.factors={AtA;A;trainExtr;trainSet};
        out.facts=size(AtA,1);
        out.optionTr=opt.option;
    case 'pca'
        %         optionDefault.option=[];
        %         opt=mergeOption(opt,optionDefault);
        A = princomp(trainSet','econ');
        A=A(:,1:opt.facts);
        trainExtr=A'* trainSet;
        out.feMethod=feMethod;
        out.factors={A;trainExtr};
        out.facts=opt.facts;
        out.optionTr=opt.option;
    otherwise
        error('Please find the correct feature extraction method or define your own.');
        
end
time=toc(tStart);
end