%% Program for K-means clustering implementation for Iris dataset
clear all;
close all;
clc;

%% Load training data
Ntrain = load('iris_dat.dat');
[rows,dim] = size(Ntrain);
K=3;
flag=K;
summation=0;
Y=0;

%% Compute a random permutation of all input vectors
R = randperm(rows);

%% Construct indicator matrix (each entry corresponds to the cluster of each point in X)
I = zeros(rows, 1);

%% Construct old centroid matrix
oldC = zeros(K, dim-1);

%% Construct centroids matrix
C = zeros(K, dim-1);

%% Take the first K points in the random permutation as the centroid
for k=1:K
    C(k,:) = Ntrain(R(k),1:dim-1);
end

%% Assignment of data points to corresponding clusters and finding new centroids of the clusters
while flag~=0
    flag=K;
    for n=1:rows
        %initialize the minimum distance
        centroid_index=1;
        mindist=norm(Ntrain(n,1:dim-1)-C(centroid_index,:),2);
        for j=1:K
            %find the closest centroid to the data point
            dist=norm(C(j,:)-Ntrain(n,1:dim-1),2);
            if dist<mindist
                centroid_index=j;
                mindist=dist;
            end
        end
        I(n)=centroid_index;
    end
    
    %Modify the centroids
    for z=1:K
        oldC(z,:)=C(z,:);
        C(z,:)=sum(Ntrain((I==z),1:dim-1));
        C(z,:)=C(z,:)/length(find(I==z));
        if norm(oldC(z,:)-C(z,:),2)<0.001
            flag=flag-1;
        end
    end
    
end

%% Validation
for n=1:rows
    centroid_index=1;
    mindist=norm(Ntrain(n,1:dim-1)-C(centroid_index,:),2);
    for j=1:K
         %find the closest centroid to the data point
         dist=norm(C(j,:)-Ntrain(n,1:dim-1),2);
         if dist<mindist
             centroid_index=j;
             mindist=dist;
         end
    end
    I(n)=centroid_index;
end

%% Construct confusion matrix
confusion_matrix = zeros(K, K);
for a=1:rows
    r_index=I(a);
    c_index=Ntrain(a,5:end);
    confusion_matrix(r_index,c_index)=confusion_matrix(r_index,c_index)+1;
end

%display confusion matrix
disp(confusion_matrix)

%% Individual efficiancy
for i=1:K
    summation=summation+max(max(confusion_matrix(:,i)));
    Y=Y+(max(max(confusion_matrix(:,i)))/(sum(confusion_matrix(:,i)))*100);
    X=sprintf('Individual efficiancy of cluster %d : %f',i, max(max(confusion_matrix(:,i)))/(sum(confusion_matrix(:,i)))*100);
    disp(X)
end

%% Average efficiacy
N=sprintf('Average efficiancy: %f', (Y/K));
disp(N)

%% Overall efficiancy
A=sprintf('Overall efficiancy: %f', (summation/rows)*100);
disp(A)