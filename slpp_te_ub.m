function [Accuracy,ProjectionMatrix,VV_f,gnd_pred] = slpp_te_ub( TrainingVector,TestingVector,gnd,gnd_test,NumOfClass,dim,DistMeasure,NumOfNeighber, rs)

Data = TrainingVector';
MeanVectorOfAllVectorTemp = mean(Data,2);
[ ~,colData ] = size(Data);

MatrixTTemp = Data - MeanVectorOfAllVectorTemp*ones(1,colData);

[rowData,colData] = size( Data );

for i = 1 : colData
    for j = 1 : colData
        Dist( i,j ) = norm( reshape(Data(:,i),rowData,1) - ...
            reshape(Data(:,j),rowData,1) );
    end
end
beta = mean( mean(Dist,1),2 ); 
beta = beta * beta;
        
TempData = zeros(rowData,1);

for i = 1 : NumOfClass
    tempInd = find(gnd==i);
    tempProb(i) = size(tempInd,2)/size(gnd,2);
end

for i = 1 : colData
%    Fl_i = floor( (i-1)/NumOfTraining );
    for j = i : colData
%        Fl_j = floor( (j-1)/NumOfTraining );
        TempData = Data(:, i) - Data(:, j);
        if gnd(i) == gnd(j)
            Correlation(i,j) = exp(-TempData'*TempData/beta) * (1 + exp(-TempData'*TempData/beta))* (tempProb(gnd(i))*tempProb(gnd(j))); %TempData'*TempData;
        else
            Correlation(i,j) = exp(-TempData'*TempData/beta) * (1 - exp(-TempData'*TempData/beta))* (tempProb(gnd(i))*tempProb(gnd(j)));
        end
        Correlation(j,i) = Correlation(i,j);
    end
end


Similarity = zeros(colData);
TempDist = zeros(rowData,1);
for i = 1 : colData
    TempDist = Dist( :,i );
    [TempSort,IndexOfSort] = sort(TempDist);
    for j = 1 : NumOfNeighber
        Similarity(i,IndexOfSort(j)) = Correlation(i,IndexOfSort(j));
        Similarity( IndexOfSort(j),i ) = Correlation(i,IndexOfSort(j));
    end
end
clear Correlation;
D = zeros(colData);
TempD = sum(Similarity);
for i = 1 : colData
    D(i,i) =  TempD(i);
end

Similarity = pinv(D) * Similarity;
Similarity = D^(-1/2)*Similarity*D^(-1/2);

D = eye(colData);

L = D - Similarity;
options = [];
%[ProjectionMatrix] = nl_ldp_ub( TrainingVector,gnd,NumOfNeighber,rs,NumOfClass );
[ProjectionMatrix, ~] = LPP(Similarity, options, TrainingVector);


% compute accuracy

for redim =  2:30
    error = 0;
    gnd_pred = gnd_test;
    MatrixProjectionW =  ProjectionMatrix(:, 1 : redim);
    RedimTrain = MatrixProjectionW'* TrainingVector';
    RedimTest = MatrixProjectionW'* TestingVector';
    [rowT,colT] = size(RedimTest);
    MeanOfClass = zeros(rowT,NumOfClass);
    CovOfClass = [];
    if strcmp(DistMeasure, 'mindist')
        for i = 1 : NumOfClass
            TempInd = find(gnd == i);
            MeanOfClass(:,i) = mean(RedimTrain(:,TempInd),2);
            CovOfClass(:,:,i) = cov(RedimTrain(:,TempInd)');
        end
        
        for i = 1 : colT
            TempM = RedimTest(:,i)*ones(1,NumOfClass) - MeanOfClass;
            for j = 1 : NumOfClass
                TempV(j) = squeeze(TempM(:,j))'* squeeze(TempM(:,j));
                % bayesian T^2
                TempCov = CovOfClass(:,:,j);
                TempV(j) =  0.5 * squeeze(TempM(:,j))'*pinv(TempCov)*squeeze(TempM(:,j))+log(det(TempCov));
                VV(i,j) = TempV(j);
            end

            
            [Tempsort,SInd] = sort(TempV,'ascend');
            if SInd(1) ~= gnd_test(i)
                error = error + 1;
                gnd_pred(i) = SInd(1);
            end
        end
    else
    end
    Accuracy(redim) = 1 - error/size(gnd_test,2);
    
end
for i = 1 : size(VV,1)
    temp = - VV(i,:);
    temp = (temp-min(temp));
    VV_f(i,:) = temp/sum(temp);
end
            