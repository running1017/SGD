% make data
epoch = 500;
datanum = 100;
batch = 100;
xWidth = 20;
x = (rand(1,datanum)-0.5)*xWidth*2; yfunc = @(x) x + randn(1,datanum)*0.01;
y = yfunc(x);

% objective function
Ei = @(w1,w2,x,y) ((y-w1.*atan(w2.*x)).^2)/datanum; % w1,w2 -> matrix  x,y -> scalar
dEidw1 = @(w1,w2,x,y) (-2*y*atan(w2*x)+2*w1*atan(w2*x)^2)/datanum; % w1,w2,x,y -> scalar
dEidw2 = @(w1,w2,x,y) (-2*y*x*w1/(1+w2^2*x^2)+w1^2*2*atan(w2*x)*x/(1+w2^2*x^2))/datanum; % w1,w2,x,y->scalar
E = @(w1,w2,x,y) sumfunc(Ei, w1, w2, x, y); % w1,w2 -> matrix  x,y -> vector
dEdw1 = @(w1,w2,x,y) sumfunc(dEidw1, w1, w2, x, y); %w1,w2 -> scalar  x,y -> vector
dEdw2 = @(w1,w2,x,y) sumfunc(dEidw2, w1, w2, x, y); %w1,w2 -> scalar  x,y -> vector

% draw surface
[w1,w2] = meshgrid(-20:0.2:20);
surfc(w1, w2, E(w1, w2, x, y), 'EdgeColor', 'interp', 'FaceColor', 'interp', 'FaceAlpha', 0.5);
hold on
title({['y=' func2str(yfunc)], ['E=' func2str(Ei)]}, 'FontSize', 20);
xlabel('w1', 'FontSize', 15); ylabel('w2', 'FontSize', 15);
axis manual
fig = gcf;
fig.Position = [100, 100, 800, 600];
view(-30,60);

algname = {'SGD', 'MOM', 'NAG', 'AdaGrad', 'AdaDelta', 'RMSProp', 'Adam'};
alg = 7;

h3d = cell(1,alg); % animatedline
p = cell(1,alg); % current point
t = cell(1,alg); % algorithm text
c = [1,   0,   0; % algorithm color
     0,   0,   1;
     0.5, 0.5, 0;
     0.5, 0.5, 0.5;
     0,   1,   1;
     1,   0,   1;
     0.5, 0, 0.5];
w1 = zeros(alg, epoch*ceil(datanum/batch)+1);
w2 = zeros(alg, epoch*ceil(datanum/batch)+1);
init = [ones(alg,1)*(-1), ones(alg,1)*(1)]; %initial position
textmargin = 0.1;

% initial setting of graph
for i = 1:alg
    h3d{i} = animatedline('Color', c(i,1:3), 'Linewidth', 2);
    p{i} = plot3(init(i,1), init(i,2), E(init(i,1), init(i,2), x, y)+0.01, ...
        'o', 'MarkerFaceColor', c(i,1:3), 'DisplayName', algname{i});
    t{i} = text(init(i,1)+textmargin, init(i,2)+textmargin, E(init(i,1), init(i,2), x, y), algname{i}, ...
        'EdgeColor', c(i,1:3), 'FontSize', 15);
end

index = zeros(epoch, datanum);
for i = 1:epoch
    index(i, 1:datanum) = randperm(datanum);
end

%%% Simple Gradient Descent %%%
eta1 = @(t) 0.01;
[w1(1,:),w2(1,:)] = ...
    sgd(init(1,1), init(1,2), x, y, dEdw1, dEdw2, eta1, batch, index);

%%% Momentum %%%
eta2 = @(t) 0.01;
gamma2 = 0.9;
[w1(2,:),w2(2,:)] = ...
    mom(init(2,1), init(2,2), x, y, dEdw1, dEdw2, eta2, gamma2, batch, index);

%%% Nesterov accelerated gradient %%%
eta3 = @(t) 0.01;
gamma3 = 0.9;
[w1(3,:),w2(3,:)] = ...
    nag(init(3,1), init(3,2), x, y, dEdw1, dEdw2, eta3, gamma3, batch, index);

%%% AdaGrad %%%
eta4 = @(t) 0.1;
vareps4 = 1e-8;
[w1(4,:),w2(4,:)] = ...
    adagrad(init(4,1), init(4,2), x, y, dEdw1, dEdw2, eta4, vareps4, batch, index);

%%% AdaDelta %%%
gamma5 = 0.9;
vareps5 = 1e-6;
[w1(5,:),w2(5,:)] = ...
    adadelta(init(5,1), init(5,2), x, y, dEdw1, dEdw2, gamma5, vareps5, batch, index);

%%% RMSProp %%%
eta6 = @(t) 0.01;
gamma6 = 0.9;
vareps6 = 1e-5;
[w1(6,:),w2(6,:)] = ...
    rmsprop(init(6,1), init(6,2), x, y, dEdw1, dEdw2, eta6, gamma6, vareps6, batch, index);

%%% Adam %%%
eta7 = @(t) 0.01;
beta1_7 = 0.9; beta2_7 = 0.999;
vareps7 = 1e-8;
[w1(7,:),w2(7,:)] = ...
    adam(init(7,1), init(7,2), x, y, dEdw1, dEdw2, eta7, beta1_7, beta2_7, vareps7, batch, index);

% testplot
testfig = figure();
testfig.Position(1:2) = [900, 100];
xtest = -xWidth:0.1:xWidth;
ytest = @(w1,w2) w1.*atan(w2.*xtest);
ptest = cell(1,alg);
scatter(x, y,'x', 'DisplayName', 'Data Point');
hold on
for i = 1:alg
    ptest{i} = plot(xtest, ytest(w1(i,1),w2(i,1)), 'Color', c(i,:), 'DisplayName', algname{i});
end
axis manual
lgd = legend('show');
lgd.FontSize = 13;
lgd.Location = 'northwest';

% make UItable
tablefig = figure('Position', [900, 550, 0, 0]);
table = uitable('Parent', tablefig, 'FontSize', 25, 'BackgroundColor', [1,1,1;1,1,1;1,1,1;c]);
table.Data = makeTable(1, epoch, w1(1:alg,1), w2(1:alg,1), E(w1(1:alg,1), w2(1:alg,2), x, y), alg, algname, datanum, batch);
table.ColumnWidth = {200,200,200,200,100};
table.Position = [10, 10, table.Extent(3:4)];
tablefig.Position(3:4) = table.Extent(3:4)+20;

% Click the UItable to start drawing
waitforbuttonpress

% drawing
for k=1:epoch*ceil(datanum/batch)
    for i=1:alg
        addpoints(h3d{i},w1(i,k),w2(i,k),E(w1(i,k),w2(i,k), x, y))
        p{i}.XData = w1(i,k); p{i}.YData = w2(i,k); p{i}.ZData = E(w1(i,k),w2(i,k), x, y)+0.01;
        t{i}.Position = [w1(i,k)+textmargin, w2(i,k)+textmargin, E(w1(i,k),w2(i,k), x, y)];
        table.Data = makeTable(ceil(k*batch/datanum), epoch, w1(1:alg,k), w2(1:alg,k), E(w1(1:alg,k), w2(1:alg,k), x, y), alg, algname, datanum, batch);
        ptest{i}.YData = ytest(w1(i,k), w2(i,k));
    end
    
    %pause(10/(k+20));
    drawnow
end

function e = sumfunc(fi, x, y, w1, w2)
    e = fi(x, y, w1(1), w2(1));
    if length(w1) > 1
        for i = 2:length(w1)
            e = e + fi(x, y, w1(i), w2(i));
        end
    end
end

function d = makeTable(k, n, x, y, z, alg, algname, datanum, batch)
    d = {'epoch', [num2str(k) '/' num2str(n)], [], [];
         'datanum', num2str(datanum), 'batch', num2str(batch);
         'alg name', 'w1', 'w2', 'E(w1,w2)'};
    for i = 1:alg
        d(i+3,1:4) = {algname{i}, x(i), y(i), z(i)};
    end
end