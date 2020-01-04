% Gain function for use with LCA - scaled by threshold and negative values
% set to zero
%
% input = activation to be transformed
% g = gain (1/g = standard deviation of cumulative normal function)

function output = gain_f_lca_DecNoise_fast(input,g,theta)

output = -theta+(2.*theta.*(normcdf(input,0,1/g)-(ones(size(input)).*normcdf(-theta,0,1/g)))./((ones(size(input)).*normcdf(theta,0,1/g))-(ones(size(input)).*normcdf(-theta,0,1/g))));


