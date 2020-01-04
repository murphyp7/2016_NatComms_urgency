% Function for simple logistic regression fitting
%
% params(1) = model intercept
% params(2) = slope
%
% data(:,1) = IV
% data(:,2) = DV

function ssr = logistic_fit(params,data)

b0 = params(1);
b1 = params(2);

for i=1:size(data,1);
    pred(i,1) = 1./(1+exp(-(b0+(b1*data(i,1)))));  % calculating model predictions
end

ssr = sum((data(:,2)-pred).^2);