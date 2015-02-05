function [testExtr,outTest,time]=featureExtrationTest(trainSet,testSet,outTrain)
% map the test/unknown data into the feature space produced by function "featureExtractionTrain".
% testSet: matrix, each column is a test/unknown sample.
% outTrain: the output of "featureExtractionTrain" function.
% testExtr: the test/unknown samples in the feature space.
% outTest: [], the other outputs, reserved for future use.
% Reference:
%  [1]\bibitem{bibm10}
%     Y. Li and A. Ngom,
%     ``Non-negative matrix and tensor factorization based classification of clinical microarray gene expression data,''
%     {\it IEEE International Conference on Bioinformatics \& Biomedicine},
%     2010, pp.438-443.
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

tStart=tic;
switch outTrain.feMethod
    case 'none'
        testExtr=testSet;
        outTest=[];
    case 'nmf'
        outTrain=rmfield(outTrain,'feMethod');
        testExtr=nmfnnlstest(testSet,outTrain);
        outTest=[];
    case 'sparsenmf'
        outTrain=rmfield(outTrain,'feMethod');
        testExtr=sparsenmfnnlstest(testSet,outTrain);
        outTest=[];
    case 'orthnmf'
        outTrain=rmfield(outTrain,'feMethod');
        testExtr=orthnmfruletest(testSet,outTrain);
        outTest=[];
    case 'knmf-nnls'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
        testExtr=kfcnnls2(AtA,AtS);
        outTest=[];
    case 'knmf-l1nnls'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
%         lambdaTlambda=outTrain.optionTr.lambda^2 .* ones(outTrain.facts);
        if ~isfield(outTrain.optionTr,'lambda')
            outTrain.optionTr.lambda=0.1;
        end
        testExtr=kfcnnls2(AtA+outTrain.optionTr.lambda,AtS);
        outTest=[];
    case 'knmf-l1nnqp'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
        if ~isfield(outTrain.optionTr,'lambda')
            outTrain.optionTr.lambda=0.1;
        end
        testExtr=l1NNQPActiveSet(AtA,-AtS,outTrain.optionTr.lambda);
        outTest=[];
    case 'knmf-ur'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        StS=computeKernelMatrix(testSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
        testExtr=kurnnls(StS,AtA,AtS);
        outTest=[];
    case 'knmf-dc'
        % the input is not testSet, it is a kernel matrix
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        [testExtr,~,~]=nmfnnlstest(XtS,outTrain);
        outTest=[];
    case 'knmf-cv'
        AtA=outTrain.factors{1};
        W=outTrain.factors{2};
        trainSet=outTrain.factors{4};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=W'*XtS;
        testExtr=kfcnnls2(AtA,AtS);
        outTest=[];
    case 'ksr-nnls'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
        StS=computeKernelMatrix(testSet,testSet,outTrain.optionTr);
        StS=diag(StS);
        % sparse coding
        [testExtr,~,sparsity]=KSRSC(AtA,AtS,StS,outTrain.optionTr);
        outTest=[];
    case 'ksr-l1nnls'
        AtA=outTrain.factors{1};
        trainExtr=outTrain.factors{2};
        XtS=computeKernelMatrix(trainSet,testSet,outTrain.optionTr);
        AtS=pinv(trainExtr)'*XtS;
        StS=computeKernelMatrix(testSet,testSet,outTrain.optionTr);
        StS=diag(StS);
        % sparse coding
        [testExtr,~,sparsity]=KSRSC(AtA,AtS,StS,outTrain.optionTr);
        outTest=[];
    case 'seminmf'
        outTrain=rmfield(outTrain,'feMethod');
        testExtr=seminmfnnlstest(testSet,outTrain);
        outTest=[];
    case 'convexnmf'
        outTrain=rmfield(outTrain,'feMethod');
        testExtr=convexnmfruletest(testSet,outTrain);
        outTest=[];
    case 'transductive-nmf'
        testExtr=outTrain.factors{4};
        outTest=[];
    case 'vsmf'
        option=outTrain.optionTr;
        AtA=outTrain.factors{1};
        A=outTrain.factors{2};
        trainExtr=outTrain.factors{3};
        
%         if option.kernelizeAY~=1 && option.alpha1>0
%             AtS=A'*S;
%         end
        if option.kernelizeAY==1 || (~option.t1 && option.alpha1==0)
            XtS=computeKernelMatrix(trainSet,testSet,option);
            I1=option.alpha2.*eye(outTrain.facts);
            H1=trainExtr*trainExtr'+I1;
            Ydagger=trainExtr'/H1;
            AtS=Ydagger'*XtS;
        else
            AtS=A'*testSet;
        end
        I2=option.lambda2.*eye(outTrain.facts);
        H2=AtA+I2;
        if option.kernelizeAY~=2 && option.t2
            G2=option.lambda1-AtS;
            testExtr=NNQPActiveSet(H2,G2);
            testExtr=max(testExtr,0);
        end
        if option.kernelizeAY~=2 && ~option.t2 && option.lambda1>0
            G2=-AtS;
            testExtr=l1QPActiveSet(H2,G2,option.lambda1);
        end
        if option.kernelizeAY==2 || (~option.t2 && option.lambda1==0)
            Adagger=H2\A';
            testExtr=Adagger*S;
        end
        outTest=[];
    case 'pca'
        testExtr=(outTrain.factors{1})'*testSet;
        outTest=[];
end
time=toc(tStart);
end