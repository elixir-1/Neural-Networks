                        %%%RBF AS A CLASSIFIER%%%
close all;
clear all;
clc;

data = xlsread('BERK7525_60.xlsx');
k=11;                                  
%cen=zeros(3,4);                       %Initialize the centroids with zero
[m,n]=size(data);    
arr=zeros(m,1);
n=n-10;
[idx,c]=kmeans(data(:,1:n-1),k);

max =0;
for i=1:k  
    for j=1:k
        dist = abs(norm(c(i,:)-c(j,:)));
        if(max<dist)
            max = dist;
        end
    end
end
spread = max/sqrt(k)
%output matrix
g = zeros(m,k);
%sigma = (-k/(2*max*max));
for i=1:m
    for j=1:k
        g(i,j) = exp(-(norm(data(i,1:(n-1))-c(j,:)).^2)/(2*spread*spread));
    end
end
g1 = pinv(g);
d= zeros(m,1);
l=1;
for i=1:m
    class=0;
    for j=n:n+10
        if data(i,j)==1
            class=j-n+1;
        end
    end
    data(i,n)=class;
end

weight = g1*data(:,n);
mult = g*weight
mul=round(mult);

confusion=zeros(k,k);
for i=1:m
    if mul(i)<=1
        mul(i)=1;
    end
    if mul(i)>=11
        mul(i) = 11;
    end
end
for i=1:m
        confusion(data(i,n),mul(i))=confusion(data(i,n),mul(i))+1;
end
confusion
sum=0;
sum1=0;
sum2=0;
for i=1:k
    max =0;
    for j=1:k
        sum2=sum2+confusion(i,j);
    if max<confusion(i,j)
        max = confusion(i,j);
    end
    end
    ind_e = max/sum2
    sum = sum+ind_e;
    sum1  = sum1+max;
    sum2=0;
end
average = sum/k
overall = sum1/m

%% Testing the network
Ntest=xlsread('BERK_test_s60.xlsx');
[r,co] = size(Ntest);
co=co-10;
for i=1:r
    for j=1:k
        g(i,j) = exp(-(norm(Ntest(i,1:(co-1))-c(j,:)).^2)/(2*spread*spread));
    end
end
g1 = pinv(g);
d= zeros(r,1);
l=1;
for i=1:r
    class=0;
    for j=co:co+10
        if Ntest(i,j)==1
            class=j-co+1;
        end
    end
    Ntest(i,co)=class;
end

weight = g1*Ntest(:,co);
mult = g*weight
mul=round(mult);
fileID = fopen('BERK_RBF_ouput.txt','w');
fprintf(fileID,'%f\n',mul);
fclose(fileID);