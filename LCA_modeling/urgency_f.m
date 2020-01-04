% Creates logistic urgency function:
%
% u0 = urgency @ t=0
% lambda = scale parameter for time of urgency increase
% k = shape parameter for shape of time-dependent urgency
% tmax = maximum time over which to chart urgency function (in seconds)
% stepsize = size of time-steps of charted function (in seconds)

function [urg,ts,urg_d1] = urgency_f(u0,lambda,k,tmax,stepsize,dir)

urg=[]; urg_d1=[];
for t = 0:stepsize:tmax;
    if strcmp(dir,'up')
        urg(end+1) = u0+(1-exp(-(t/lambda).^k));
        urg_d1(end+1) = (k*((t/lambda)^k)*(exp(-(t/lambda)^k)))/t;
    elseif strcmp(dir,'down')
        urg(end+1) = u0-(1-exp(-(t/lambda).^k));
        urg_d1(end+1) = (-k*((t/lambda)^k)*(exp(-(t/lambda)^k)))/t;
    end
end
ts=0:stepsize:tmax;