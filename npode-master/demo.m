% updating the path
addpath(genpath('.'))

%%
% data generation
x0 = [2 0]; % initial value

model = 'sp'; % Van der Pol oscillator as a toy model
[t,Y] = gen_data('model',model, 'sn', 0.0001, 'x0', x0, 'Ny', 20); % time points and noisy data using VDP

%%

% fitting
gp = npode_fit(t,Y, 'optpars', 'log_sn-Fw', 'W', 4, 'model', model); % fits npODE model and returns the learned parameters
% lengthscale and ind. points grid width can be set optionally:
% gp = npode_fit(t,Y,'W',6,'ell',[2 2]); 
gp.x0_true = x0;
% visualization
gp.ode_model = model; % needed to visualize true states
plotmodel(gp) % plots the vector field, true states and trajectories

%

% predicting future cycles
ts = 0:0.01:3; % new time points
X = npode_predict(gp,ts,x0); % prediction

errorts = 0:0.07894736842105263:3;
errorX = npode_predict(gp,errorts,x0);

cols = ggplotcolors(2);
figure(1)
subplot(2,2,2);
hold on
plot(ts, X(:,1), 'color', cols(1,:));
plot(ts, X(:,2), 'color', cols(2,:));
hold off


% predicting longer circle
longts = 0:0.01:25; % new time points
longX = npode_predict(gp,longts,x0); % prediction




% reduce noise (or rather, find option to set it)
% extract inducing points, inducing point values, kernel parameters

%%
Zh = gp.X;
Uh = gp.F;
tsh = ts;
tpredh = X;
uncerth = gp.sn;
kernelparh = [gp.sf, gp.ell];
longtsh = longts;
longtpredh = longX;
errortsh = errorts;
errorXh = errorX;


save('../data/heinonen_data.mat', 'Zh', 'Uh', 'tsh', 'tpredh', 'uncerth', 'kernelparh', 'longtsh', 'longtpredh', 'errortsh', 'errorXh');



