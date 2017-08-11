close;
clear all;
%happen=160;
 
%% read in data including normal and fault data
d00=importdata('d00.dat');
d00te = importdata('d00_te.dat');
% read in the training and testing data
for i = 1 : 21
    str_i = num2str(i);
    if i < 10
        str_i = strcat('0',str_i);
    end
    file_name = strcat('d',str_i,'.dat');
    file_name_test = strcat('d',str_i,'_te.dat');
    train_fault(:,:,i) = importdata(file_name);
    test_fault(:,:,i) = importdata(file_name_test);
end

train_normal=d00';
[train_normal,mean,std]=zscore(train_normal);
test_normal(:,:,i)=(d00te-ones(size(d00te,1),1)*mean)./(ones(size(d00te,1),1)*std);
for i = 1 : 21
    temp_train_fault = train_fault(:,:,i);
    train_fault(:,:,i)=(temp_train_fault-ones(size(temp_train_fault,1),1)*mean)./(ones(size(temp_train_fault,1),1)*std);
    temp_test_fault = test_fault(:,:,i);
    test_fault(:,:,i)=(temp_test_fault-ones(size(temp_test_fault,1),1)*mean)./(ones(size(temp_test_fault,1),1)*std);
end

%% Trainingset and Testingset


gnd_tr = [];
gnd_test_other = [];
tempTrain = [];
tempTest = [];
j=1;
for i =  [1,2,6,7,8]
    gnd_test_other = [gnd_test_other,[ j*ones(1,960-160)]];
    gnd_tr = [gnd_tr,j*ones(1,480)];
    tempTrain = [tempTrain; train_fault(:,:,i)];
    tempTest = [tempTest;test_fault(161:960,:,i)];
    j=j+1;
end 
Train = tempTrain;
Test = tempTest;


[Accuracy,ProjectionMatrix,VV] = slpp_te_ub( Train,Test,gnd_tr,gnd_test_other,5,30,'mindist',30, 30);
plot(Accuracy,'ro-');
legend('NDGE');
xlabel('Dimension')
ylabel('Accuracy')

