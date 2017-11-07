%% Forcasting using RBF
clear all;
close all;
clc;

%% Load training data
Ntrain = xlsread('SI_23.xlsx');
[rows, cols] = size(Ntrain)
Ntrain=[ones(rows,1) Ntrain];
[m,n]=size(Ntrain)
for i=2:3
    Ntrain(:,i)=(Ntrain(:,i)-min(Ntrain(:,i)))/(max(Ntrain(:,i))-min(Ntrain(:,i)));
end

%% Initialize the algorithm parameters
input = 2;
hidden_neurons = 5;
output = 1;
learning_rate = 1.e-05;
epo=100;

%% Hidden output matrix
hidden_output = zeros(rows,hidden_neurons);

%% Weight matrix
W = zeros(hidden_neurons, 1);

%% Initialize the centres
R = randperm(rows);
C = zeros(hidden_neurons, input);
for k=1:hidden_neurons
    C(k,:) = Ntrain(R(k),1:input);
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
    DW = zeros(output, hidden_neurons);
    DC = zeros(hidden_neurons, input);
    for i=1:rows
        for j=1:hidden_neurons
            hidden_output(i,j) = exp(-(norm(Ntrain(i,1:input)-C(j,:)).^2)/(2*Spread*Spread));
        end
    end
    W = pinv(hidden_output)*Ntrain(:,input+1);
end

%% Validate the network
rmstra = zeros(output,1);
res_tra = zeros(rows,2);
for i = 1:rows
    for j=1:hidden_neurons
        hidden_output(i,j) = exp(-(norm(Ntrain(i,1:input)-C(j,:)).^2)/(2*Spread*Spread));
    end
end
Actual_output=Ntrain(:,input+1);
Predicted_output=hidden_output*W;
rmstra = rmstra + (Actual_output-Predicted_output).^2;
%res_tra(i,:) = [tt Yo];
disp(mean(sqrt(rmstra/rows)))

%% Test the network
Ntest = xlsread('SI_test_s23.xlsx');
[r,c] = size(Ntest);
Ntest=[ones(r,1) Ntest];
for i=2:3
    Ntest(:,i)=(Ntest(:,i)-min(Ntest(:,i)))/(max(Ntest(:,i))-min(Ntest(:,i)));
end
phidden_output=zeros(r,hidden_neurons);
prediction = zeros(r,1);
for i = 1:r
    for j=1:hidden_neurons
        phidden_output(i,j) = exp(-(norm(Ntest(i,1:input)-C(j,:)).^2)/(2*Spread*Spread));
    end
end
Predicted_output=phidden_output*W;
%disp('Prediction')
%disp(Predicted_output)

fileID = fopen('SI_ouput_rbf.txt','w');
fprintf(fileID,'%f\n',Predicted_output);
fclose(fileID);
