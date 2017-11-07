%% Approximation using RBF using Pseudo inverse method
clear all;
close all;
clc;

%% Load training data
Ntrain = xlsread('SI_23.xlsx');
[rows, dim] = size(Ntrain);

%% Initialize the algorithm parameters
input = 2;
hidden_neurons = 5;
output = 1;
learning_rate = 1.e-06;
epo=100;

%% Hidden output matrix
hidden_output = zeros(rows,hidden_neurons);

%% Weight matrix
W = zeros(hidden_neurons, 1);

%% Initialize the centres
R = randperm(rows);
%[~,C]=kmeans(Ntrain(:,1:n-11),k);
C = zeros(hidden_neurons, dim-1);
for k=1:hidden_neurons
    C(k,:) = Ntrain(R(k),1:dim-1);
end

%% Calculate Spread
dmax=0;
for i=1:k  
    for j=1:k
        dist = abs(norm(C(i,:)-C(j,:)));
        if(dmax<dist)
            dmax = dist;
        end
    end
end
Spread = dmax/sqrt(hidden_neurons);

%% Training the network
for epoch=1:epo
    error_sum=0;
    for i=1:rows
        for j=1:hidden_neurons
            hidden_output(i,j) = exp(-(norm(Ntrain(i,1:dim-1)-C(j,:)).^2)/(2*Spread*Spread));
        end
    end
    W = pinv(hidden_output)*Ntrain(:,dim);
end

%% Validate the network
rmstra = zeros(output,1);
res_tra = zeros(rows,2);
for i = 1:rows
    for j=1:hidden_neurons
        hidden_output(i,j) = exp(-(norm(Ntrain(i,1:dim-1)-C(j,:)).^2)/(2*Spread*Spread));
    end
end
Actual_output=Ntrain(:,dim);
Predicted_output=hidden_output*W;
rmstra = rmstra + (Actual_output-Predicted_output).^2;
%res_tra(i,:) = [tt Yo];
disp(sqrt(rmstra/rows))
disp(mean(sqrt(rmstra/rows)))
%% Test the network
%NFeature=load('her.tes');
%[NTD,~]=size(NFeature);
%rmstes = zeros(output,1);
%res_tes = zeros(NTD,2);
%for i = 1:rows
%    for j=1:hidden_neurons
%        hidden_output(i,j) = exp(-(norm(Ntrain(i,1:dim-1)-C(j,:)))/2*Spread^2);
%    end
%end
%Actual_output=Ntrain(:,dim);
%Predicted_output=hidden_output*W;
%rmstes = (Actual_output-Predicted_output).^2;
%res_tes = [Actual_output Predicted_output];
%disp(sqrt(rmstes/NTD))