%% Program for implementation of Self Organizing Map
clear all;
close all;
clc;

%% Load training data
Ntrain = load('iris_dat.dat');
[rows,dim] = size(Ntrain);

%% Initialize the Algorithm Parameters.....................................
inp = 4;                      % No. of input neurons
out = 3;                      % No. of Output Neurons
l = 1;                        % Learning rate
epo = 100;                    % No. of iterartions
D = zeros(rows,out+1);        % Distance matrix
epoch=0;
flag=1;
lambda=4;                     % For exponential decay of learning rate
w_neighborhood = 2;           % Width of neighborhood
variance = w_neighborhood.^2; % Variance

summation=0;
Y=0;

%% Initialize the weights
W = zeros(out,inp);
for j=1:dim-1
    minimum=min(Ntrain(:,j));
    maximum=max(Ntrain(:,j));
    W(:,j) = (minimum+(maximum-minimum)*rand(out,1));
end

%% Construct neighborhood_function matrix
neighborhood_function = zeros(out,1);

%% SOM iterations
for epoch=1:epo
%while flag~=0
    l=l/(epoch+1);
    %w_neighborhood=w_neighborhood*exp(-lambda*epoch);
    for i=1:rows
        min =  norm(W(1,:)-Ntrain(i,1:dim-1),2);
        D(i,out+1)=1;
        for j = 1:out
            D(i,j) = norm(W(j,:)-Ntrain(i,1:dim-1),2);
            if(D(i,j)<min)
                min = D(i,j);
                D(i,out+1)=j;
            end
        end
           
        % Compute neighborhood
        for j=1:out
            winning_neuron = D(i,out+1);
            distance = abs(winning_neuron-j);
            neighborhood_function(i,:)=exp(-distance/variance);
        end
        % Update the weights
        old_W=W;
        for j=1:out
            W(j,:)=W(j,:)+l.*neighborhood_function(j,1).*(Ntrain(i,1:dim-1)-W(j,:));
        end
   
        if norm(old_W-W)<0.001
            flag=0;
        end
        
    end
end

disp(D)

%% Construct confusion matrix
confusion_matrix = zeros(out, out);
for a=1:rows
    r_index=D(a,out+1);
    c_index=Ntrain(a,5:end);
    confusion_matrix(r_index,c_index)=confusion_matrix(r_index,c_index)+1;
end

%display confusion matrix
disp(confusion_matrix)

%% Individual efficiancy
for i=1:out
    summation=summation+max(max(confusion_matrix(:,i)));
    Y=Y+(max(max(confusion_matrix(:,i)))/(sum(confusion_matrix(:,i)))*100);
    X=sprintf('Individual efficiancy of cluster %d : %f',i, max(max(confusion_matrix(:,i)))/(sum(confusion_matrix(:,i)))*100);
    disp(X)
end

%% Average efficiacy
N=sprintf('Average efficiancy: %f', (Y/out));
disp(N)

%% Overall efficiancy
A=sprintf('Overall efficiancy: %f', (summation/rows)*100);
disp(A)

%% Plot
subplot(1,2,1);
scatter(Ntrain(:,1),Ntrain(:,3),20,D(:,out+1),'filled');
hold on;
subplot(1,2,2);
scatter(Ntrain(:,1),Ntrain(:,3),20,Ntrain(:,5),'filled');
hold on;