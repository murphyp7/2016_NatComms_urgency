function [ML] = ML_from_LCA_FR(pm)

% Retrieving global variables from initializing script
global nsims
global i0 Ter betaM effleak g0
global Qc Qe Oc Oe

% Calculating fpt densities for each particle
for particle = 1:size(pm,1);
    
    % Pull params for each condition
    pm_i = [pm(particle,:)...                                       % parameter set: [lambda v theta noise eta glambda gk...
        i0 g0 Ter...                                                %                 i0 g0 Ter...
    (pm(particle,1)-effleak)/(1+betaM)...                           %                 alpha...
    (pm(particle,1)-effleak)/(1+betaM)*betaM];                      %                 beta];
        
    % Simulate behaviour & sorting simulated RTs by accuracy
    [RT,ACC] = lca_sim_gain_FR(pm_i,nsims);
    
    RTc = RT(ACC==1);
    RTe = RT(ACC==0);
    misses = length(find(isnan(RT)));
    
    % Calculating quantile likelihoods
    Lc=[]; Le=[];
    for q = 1:length(Qc) % corrects
        if q==1
            Lc(q) = length(find(RTc<=Qc(1)))/nsims;
        else Lc(q) = length(find(RTc>Qc(q-1) & RTc<=Qc(q)))/nsims;
        end
    end
    Lc(end+1) = (length(find(RTc>Qc(end)))+(misses/2))/nsims;  % final quantile - splitting misses (if any) evenly between corrects/errors (any sims that have misses @ maxt=6s will fit poorly anyway, so doesn't matter)
    
    for q = 1:length(Qe) % errors
        if q==1
            Le(q) = length(find(RTe<=Qe(1)))/nsims;
        else Le(q) = length(find(RTe>Qe(q-1) & RTe<=Qe(q)))/nsims;
        end
    end
    Le(end+1) = (length(find(RTe>Qe(end)))+(misses/2))/nsims;
    
    if length(find(Lc==0))>0, Lc(Lc==0) = 1e-10; end  % setting zero likelihood to very small real values to avoid -Inf likelihoods
    if length(find(Le==0))>0, Le(Le==0) = 1e-10; end
    
    for q = 1:length(Lc), Lc(q)=(-log(Lc(q)))*Oc(q); end  % converting to negative log-likelihoods and scaling by number of observations for this quantile
    for q = 1:length(Le), Le(q)=(-log(Le(q)))*Oe(q); end
    
    ML(particle,1) = sum(Lc)+sum(Le); % total negative log-likelihood
end