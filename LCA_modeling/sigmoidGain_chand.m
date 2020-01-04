function [V, Vratio] = sigmoidGain(x, StdParameter, Theta, varargin)
% sigmoidGain implements the nonlinearity described in Murphy et al. Nat Comm. 2016
% 
% Chandrasekaran, April 2019


type='normpdf';
assignopts(who, varargin);

if x <= -Theta
    V = -Theta;
    Vratio = 0;
end
if x >-Theta && x <=Theta
    
    if type=='normpdf'
        V1 = integral(@(y)normpdf(y,0,StdParameter),-Theta,x);
        V2 = integral(@(y)normpdf(y,0,StdParameter),-Theta,Theta);
    else
        V1 = integral(@(y)normcdf(y,0,StdParameter),-Theta,x);
        V2 = integral(@(y)normcdf(y,0,StdParameter),-Theta,Theta);
    end
    
    Vratio = V1/V2;
    
    V = -Theta + 2*Theta*(Vratio);
end
if x > Theta
    V = Theta;
    Vratio = 0;
end
end

