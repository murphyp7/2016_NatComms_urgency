% Gain function for use with LCA, scaled by threshold
%
% input = activation to be transformed
% g = gain (1/g = standard deviation of cumulative normal function)
% theta = threshold

function output = gain_f_lca(input,g,theta)

output = -theta+(2.*theta.*(normcdf(input,0,1/g)-(ones(size(input)).*normcdf(-theta,0,1/g)))./((ones(size(input)).*normcdf(theta,0,1/g))-(ones(size(input)).*normcdf(-theta,0,1/g))));

output(output>theta) = theta;
output(output<-theta) = -theta;


