% spiral ODE system
%
% to run:
%  [ts,V] = ode45(@(t,y) vdp(t,y),ts,[-1 1]);
%
function dx = ode_sp(t,x)

% default initial value
if ~exist('x','var')
    x = [2; 0];
end

true_A = [-0.1 2.0; -2.0 -0.1];
dx = ((x.^3)'*true_A)';

%	dx = [x(2);
%		 (1-x(1).^2).*x(2)-x(1)];
%      dx = [x(:,2) (1-x(:,1).^2).*x(:,2)-x(:,1)];
end


