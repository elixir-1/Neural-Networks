%% Approximation using RBF using Gradient descent method
clear all;
close all;
clc;

%% Load training data
Ntrain = xlsread('fin_23.xlsx');
[rows, dim] = size(Ntrain);

%% Initialize the algorithm parameters
input = 2;
hidden_neurons = 5;
output = 1;
learning_rate = 2.e-05;
epo=100;

%% Hidden output matrix
hidden_output = zeros(rows,hidden_neurons);

%% Weight initialization
W = 0.001*(rand(hidden_neurons,output)*2.0-1.0);

%% Initialize the centres
R = randperm(rows);
C = zeros(hidden_neurons, input);
for k=1:hidden_neurons
    C(k,:) = Ntrain(R(k),1:input);
end

%% Calculate Spread
dmax=0;
for i=1:hidden_neurons
    dmax=max(norm(C(1,:)-C(i,:)));
end
Spread = dmax/sqrt(hidden_neurons);

%% Training the network
for epoch=1:epo
    error_sum=0;
    DW = zeros(hidden_neurons, output);
    DC = zeros(hidden_neurons, input);
    DS = 0;
    for i=1:rows
        for j=1:hidden_neurons
            hidden_output(i,j) = exp(-(norm(Ntrain(i,1:input)-C(j,:)))/2*Spread^2);
        end
        predicted_output=hidden_output(i,:)*W;
        actual_output = Ntrain(i,dim);
        err = actual_output-predicted_output;
        DW = DW+learning_rate*err*hidden_output(i,:)';
        temp=0;
        for ind=1:hidden_neurons
            temp1=(-2)*learning_rate*err*hidden_output(i,ind)*W(ind,:);
            DC(ind,:) = DC(ind,:)+(temp1*(Ntrain(i,1:input)-C(ind,:)))./Spread^2;
            temp=temp+W(ind,:)*hidden_output(i,ind)*((norm(Ntrain(i,1:input)-C(ind,:)))/Spread^3);
        end
        DS = DS+((-2)*learning_rate*err)*temp;
        error_sum=error_sum+sum(err.^2);
    end
    W = W+DW;
    C = C+DC;
    Spread = Spread+DS;
end

%% Validate the network
rmstra = zeros(output,1);
res_tra = zeros(rows,2);
for i=1:rows
        for j=1:hidden_neurons
            hidden_output(i,j) = exp(-(norm(Ntrain(i,1:input)-C(j,:)))/2*Spread^2);
        end
        predicted_output=hidden_output*W;
        actual_output = Ntrain(i,dim);
        err = actual_output-predicted_output;
        rmstra = rmstra + (err).^2;
%        res_tra(i,:) = [actual_output predicted_output];
end
disp(sqrt(rmstra/rows))

%% Test the network
%NFeature=xlsread('fin_test_s23.xlsx');
%[r,c]=size(NFeature);
%rmstes = zeros(output,1);
%res_tes = zeros(r,2);
%for sa = 1: r
%    for j=1:hidden_neurons
%        hidden_output(i,j) = exp(-(norm(Ntrain(i,1:input)-C(j,:)))/2*Spread^2);
%    end
%    predicted_output=hidden_output*W;
    %actual_output = Ntrain(i,dim);
    %err = actual_output-predicted_output;
    %rmstes = rmstes + (err).^2;
    %res_tes(sa,:) = [actual_output predicted_output];
%end
%disp(sqrt(rmstes/r))