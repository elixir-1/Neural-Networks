%% Forcasting using MLP
clear all;
close all;
clc;

%% Load training data
Ntrain = xlsread('SI_23.xlsx');
[rows, cols] = size(Ntrain);
Ntrain=[ones(rows,1) Ntrain];
for i=2:3
    Ntrain(:,i)=(Ntrain(:,i)-min(Ntrain(:,i)))/(max(Ntrain(:,i))-min(Ntrain(:,i)));
end
%% Initialize the algorithm parameters
input = 2;
hidden_neurons = 5;
output = 1;
learning_rate = 2.e-05;
epo=5000;

%% Initialize the weights
Wi = (rand(hidden_neurons,input)*2.0-1.0);     % Input weights
Wo = (rand(output,hidden_neurons)*2.0-1.0);    % Output weights
%disp(Wi)
%disp(Wo)

%% Train the network
for epoch=1:epo
    error_sum=0;
    DWi = zeros(hidden_neurons, input);
    DWo = zeros(output, hidden_neurons);
    for i=1:rows
        input_vector=Ntrain(i, 1:input);
        actual_output=Ntrain(i, input+1:end)';
        product = Wi*input_vector';
        hidden_output=1./(1+exp(-Wi*(input_vector)'));
        %disp(exp(product))
        predicted_output=Wo*hidden_output;
        error=actual_output-predicted_output;
        DWo=DWo+learning_rate*(error*hidden_output');
        DWi=DWi+learning_rate*((Wo'*error).*(hidden_output).*(1-(hidden_output)))*input_vector;
        error_sum=error_sum+sum(error.^2);
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
    disp(sqrt(error_sum/rows))
end
%disp(Wi)
%disp(Wo)

%% Validate the network
rmstra = zeros(output,1);
res_tra = zeros(rows,2);
for i=1:rows
    input_vector=Ntrain(i, 1:input);
    actual_output=(Ntrain(i, input+1:end));
    hidden_output=1./(1+exp(-Wi*(input_vector)'));
    predicted_output=Wo*hidden_output;
    rmstra = rmstra + (actual_output-predicted_output).^2;
    res_tra(i,:) = [actual_output predicted_output];
end
disp(mean(sqrt(rmstra/rows)))

%% Test the network
Ntest = xlsread('SI_test_s23.xlsx');
[r,c] = size(Ntest);
Ntest=[ones(r,1) Ntest];
for i=2:3
    Ntest(:,i)=(Ntest(:,i)-min(Ntest(:,i)))/(max(Ntest(:,i))-min(Ntest(:,i)));
end
prediction = zeros(r,1);
rmstes = zeros(output,1);
res_tes = zeros(r,2);
for i=1:r
    input_vector=Ntest(i, 1:input);
    hidden_output=1./(1+exp(-Wi*(input_vector)'));
    predicted_output=Wo*hidden_output;
    prediction(i,:)=predicted_output;
end
%disp('Prediction')
%disp(prediction)

fileID = fopen('SI_ouput.txt','w');
fprintf(fileID,'%f\n',prediction);
fclose(fileID);
