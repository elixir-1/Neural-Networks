%% Program for fuzzy c-means clustering for Image segmentation
clear all;
close all;
clc;

%% Load training data
training_file = imread('Landsat.jpg');

training_data = double( training_file(:,:,1));

final_data = [];

for i=1:size(training_file,2)
    final_data = cat(1,final_data,training_data(:,i,1));
end
[rows,dim]=size(final_data);
K=2;
flag=K;

%% Fuzziness index
k=6;

%% Construct old centroid matrix
oldC = zeros(K, dim);

%% Construct centroid matrix
C = zeros(K, dim);

%% Construct membership matrix
M = rand([rows, K]);
for j = 1:rows
      M(j, :) = M(j, :)./sum(M(j, :));      
end  

%% Construct distance matrix
D = zeros(rows,K);

%% Modifying the centroid matrix and membership matrix
while flag~=0
    flag=K;
    for m=1:K
        for n=1:dim
        sum_power=(M(:,m).^k);
        C(m,n) = sum(final_data(:,n).*sum_power)/sum(sum_power); %update centroid matrix
        end
    end
    
    for n=1:rows
        for m=1:K
            D(n,m)=norm(C(m,:)-final_data(n,:),2); %update distance matrix
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