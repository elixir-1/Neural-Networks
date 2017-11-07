%% Program for K-means clustering implementation for Image segmentation
clear all;
close all;
clc;

%% Load training data
training_file = imread('Landsat_cropped.jpg');

training_data = double( training_file(:,:,1));

final_data = [];

for i=1:size(training_file,2)
    final_data = cat(1,final_data,training_data(:,i,1));
end
[rows,dim]=size(final_data);
K=2;
flag=K;

%% Distance matrix
D = zeros(rows,2);

%% compute a random permutation of all input vectors
R = randperm(rows);

%% construct indicator matrix (each entry corresponds to the cluster
% of each point in X)
I = zeros(rows, 1);

%% construct old centroid matrix
oldC = zeros(K, dim);

%% construct centroids matrix
C = zeros(K, dim);

%% take the first K points in the random permutation as the centroid
for k=1:K
    C(k,:) = final_data(R(k),:);
end

%% Assignment of data points to corresponding clusters Finding new centroids of the clusters
while flag~=0
    flag=K;
    for n=1:rows
        %initialize the minimum distance
        centroid_index=1;
        mindist=norm(final_data(n,1:dim)-C(centroid_index,:));
        for j=1:K
            %find the closest centroid to the data point
            dist=norm(C(j,:)-final_data(n,1:dim));
            if dist<mindist
                centroid_index=j;
                mindist=dist;
            end
        end
        D(n,1)=mindist;
        D(n,2)=centroid_index;
        I(n)=centroid_index;
    end
    
    %Modify the centroids
    for z=1:K
        oldC(z,:)=C(z,:);
        C(z,:)=mean(final_data((I==z),:));
        %C(z,:)=C(z,:)/length(find(I==z));
        if norm(oldC(z,:)-C(z,:),2)<0.001
            flag=flag-1;
        end
    end
    
end

%% Setting intensitities for each cluster

m=0;
for i=1:size(training_data,1)
    for l=1:size(training_file,2)
        m=m+1
            if (D(:,2)==1)
                training_data(i,l) = 30;
            elseif (D(:,2)==2)
                training_data(i,l) = 180;
            end
    end
end
image(training_data);