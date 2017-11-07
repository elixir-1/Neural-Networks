% Program for  MLP..........................................
% Update weights for a given epoch

clear all;
close all;
clc;

% Load the training data..................................................
Ntrain=xlsread('SI_23.xlsx');
[NTD,~] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = 2;          % No. of input neurons
hid = 5;        % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 2.e-05;       % Learning rate
epo = 1000;

% Initialize the weights..................................................
Wi = (rand(hid,inp)*2.0-1.0);  % Input weights
Wo = (rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
%    disp(sqrt(sumerr/NTD))
%     save -ascii Wi.dat Wi;
%     save -ascii Wo.dat Wo;
end

% Validate the network.....................................................
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
for sa = 1: NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        rmstra = rmstra + (tt-Yo).^2;
        res_tra(sa,:) = [tt Yo];
end
disp(sqrt(rmstra/NTD))

% Test the network.........................................................
%NFeature=load('her.tes');
%[NTD,~]=size(NFeature);
%rmstes = zeros(out,1);
%res_tes = zeros(NTD,2);
%for sa = 1: NTD
%        xx = NFeature(sa,1:inp)';   % Current Sample
%        ca = NFeature(sa,end);      % Actual Output
%        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
%        Yo = Wo*Yh;                 % Predicted output
%        rmstes = rmstes + (ca-Yo).^2;
%        res_tes(sa,:) = [ca Yo];
%end
%disp(sqrt(rmstes/NTD))
