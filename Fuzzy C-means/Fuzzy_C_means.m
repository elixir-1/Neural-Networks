%% Program for fuzzy c-means clustering for Iris dataset
clear all;
close all;
clc;

%% Load training data
Ntrain = load('iris_dat.dat');
[rows,dim] = size(Ntrain);
K=3;
flag=K;
iter=100;
summation=0;
Y=0;

%% Fuzziness index
k=6;

%% Input data
data = Ntrain(:,1:dim-1);

%% Construct old centroid matrix
oldC = zeros(K, dim-1);

%% Construct centroid matrix
C = zeros(K, dim-1);

%% Construct membership matrix
M = rand([rows, K]);
for j = 1:rows
      M(j, :) = M(j, :)./sum(M(j, :));      
end  

%% Construct distance matrix
D = zeros(rows,K);

%% Modifying the centroid matrix and membership matrix
while iter~=0 || flag~=0
    iter=iter-1
    flag=K;
    for m=1:K
        for n=1:dim-1
        sum_power=(M(:,m).^k);
        C(m,n) = sum(Ntrain(:,n).*sum_power)/sum(sum_power); %update centroid matrix
        end
    end
    
    for n=1:rows
        for m=1:K
            D(n,m)=norm(C(m,:)-Ntrain(n,1:dim-1),2); %update distance matrix
        end
    end
    
    oldC = C;
    den=zeros(rows,1);
    for ro=1:rows
        d=0;
        for co=1:K
            d=d+(1/(D(ro,co).^(2/(k-1))));
        end
        den(ro)=d;
    end
    for x=1:rows 
        for y=1:K
            M(x,y) = (1/(D(x,y).^(2/(k-1))))./den(x); %update membership matrix
        end
    end
    
    for m=1:K
        if norm(oldC(m,:)-C(m,:),2)<0.01 %check for stopping condition
            flag=flag-1;
        end
    end
end

%% Construct confusion matrix
confusionMatrix = zeros(K,K);
targetOutput = Ntrain(:,end);
prediction = zeros(rows, 1);

for i=1:rows
   [~,prediction(i)]=max(M(i,:));
end

for i=1:rows
    I_row=prediction(i);
    I_col=targetOutput(i);
    confusionMatrix(I_row,I_col) = confusionMatrix(I_row,I_col)+1;
end
disp(confusionMatrix);

%% Individual efficiancy
for i=1:K
    summation=summation+max(max(confusionMatrix(:,i)));
    Y=Y+(max(max(confusionMatrix(:,i)))/(sum(confusionMatrix(:,i)))*100);
    X=sprintf('Individual efficiancy of cluster %d : %f',i, max(max(confusionMatrix(:,i)))/(sum(confusionMatrix(:,i)))*100);
    disp(X)
end

%% Average efficiacy
N=sprintf('Average efficiancy: %f', (Y/K));
disp(N)

%% Overall efficiancy
A=sprintf('Overall efficiancy: %f', (summation/rows)*100);
disp(A)