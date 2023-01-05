clc;clear all;close all;
%Data set defination
x = rand(2000,1)*5;
y = rand(2000,1)*5;
c = mod((floor(x)+floor(y)),2);
ind = find(c);
a = [x(ind),y(ind)];
ind1 = find(c==0);
b = [x(ind1),y(ind1)];

A=[a;b];
D =diag([-1.*ones(length(a),1);1.*ones(length(b),1)]);

%mapping the dataset into feature space
n=length(A);
k=[];
r=randi([0 1],5,length(A(1,:)));%Random Matrix

%kernel matrix using random kitchen sink
for i=1:n
    y=r*A(i,:)';
    z=[cos(y);sin(y)];
    k(i,:)=z;
end

%formulating the optimization problem and solving it using cvx
n=length(k(:,1));
e = ones(n,1);
c=1.2 ;
cvx_begin
    variable w(length(k(1,:)))
    variable g(1)
    variable ep(n) 
    om=(((w'*w)/2)+sum((c*ep)));
    cont=D*((k*w)-(g.*e))+ep-e;
    minimize om
    subject to 
        cont >= 0;
        ep >= 0;
cvx_end
%hyperplane parameters
w 
g
%Testing a datapoint
x=[3.5;2.5];
y=r*x;
knew=[cos(y);sin(y)];%mapping the test datapoint to feature space
s=sign((knew'*w)-g);

if s==1
    disp("class is 1")
else
    disp("class is -1 ")
end