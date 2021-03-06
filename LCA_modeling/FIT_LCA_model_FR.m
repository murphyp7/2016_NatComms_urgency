% function [pm_best,L,AIC,BIC,tr,te] = FIT_LCA_model_FR(subj)
%
% Fits the full two-choice LCA model to data from free response condition
% pooled over subjects using particle swarm optimzation.
%
% Outputs: 'pm_best' = vector of best-fitting diffusion parameters
%          'L' = likelihood value minimized by PSO procedure
%          'AIC' = Akaike Information Criterion
%          'BIC' = Bayesian Information Criterion
%          'tr' = likelihood minimum at every iteration
%          'te' = number of iterations run
%
% Peter Murphy, UKE Hamburg, 04/01/2020

function [pm_best,L,AIC,BIC,tr,te] = FIT_LCA_model_FR

% set global varaibles for passing to PSO routine
global nsims
global i0 Ter betaM effleak g0
global Qc Qe Oc Oe n_trials

% Adding required paths
addpath(genpath('/mnt/homes/home024/pmurphy/RDM_DL/LCA_modeling/Carland_serial'))

% Define file/directory info directory
loadpath = '/mnt/homes/home024/pmurphy/RDM_DL/LCA_modeling/Behaviour/';
savepath = '/mnt/homes/home024/pmurphy/RDM_DL/LCA_modeling/FITS/Carland/';

allsubj = {'06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26'};
n_blocks = 4;

% Number of simulated trials per iteration of fitting
nsims = 10000;

% Parameter constraints
i0 = 0.2;   % fixed baseline input
Ter = 0.39;  % fixed non-decision time (in seconds)
betaM = 2/3;       % excitation = 1.5 x inhibition
effleak = 0.06;  % fixed effective differential leak (lambda-alpha-beta) - 0.06 equates to the same time constant used in Carland et al. (dt/0.06 = 0.01/0.06 = 0.1667s)
g0 = 0.01;   % fixed network gain @ t=0

range_p.lamda = [0.16 0.5];  % leak
range_p.v = [0.501 0.65];    % stimulus input to correct accumulator (incorrect input will be 1-v)
range_p.theta = [2 13];      % decision bound
range_p.noise = [0.2 0.6];  % noise
range_p.eta = [0.001 0.2];   % b/w trial variability in stimulus input
range_p.glamda = [0.5 15];  % scale of gain urgency signal
range_p.gk = [0.5 15];   % shape of gain urgency signal

mv = [0.04;...    % maximum particle velocities
      0.01;...
      0.25;...
      0.06;...
      0.02;...
      0.25;...
      0.25]';

seeds.lamda=[0.25 0.06];  % good seed distributions for a selection of particles - [mean sd]
seeds.v=[0.53 0.007];
seeds.theta=[9.4 0.8];
seeds.noise=[0.45 0.08];
seeds.eta=[0.025 0.01];
seeds.glamda=[7 5];
seeds.gk=[7 5];

% Seed random number generator
seed = round(sum(100*clock)); %never the same seed
rand('state', seed);

% RT trimming threshold for FR condition
RTstd = 4;   % first discard all trials with mean+(std*this_value) RT
RTthresh = 5;  % then discard all remaining trials with RT > this value (in secs)

% Looping through subject/blocks & pooling behaviour
for p = 1:length(allsubj);
    subj = allsubj{p};
    
    for b = 1:n_blocks
        load([loadpath,subj,'_EEG/',subj,'_EEG_FR',num2str(b),'.mat'])
        if b==1
            ACC_FR = data_block(:,2);
            RT_FR = data_block(:,4);
        else ACC_FR(end+1:end+size(data_block,1)) = data_block(:,2);
            RT_FR(end+1:end+size(data_block,1)) = data_block(:,4);
        end
    end
    
    ACC_FR = ACC_FR(find(RT_FR>0)); RT_FR = RT_FR(find(RT_FR>0)); % discard misses (there shouldn't be any in this condition though)
    ACC_FR = ACC_FR(find(RT_FR<(mean(RT_FR)+(RTstd*std(RT_FR))))); RT_FR = RT_FR(find(RT_FR<(mean(RT_FR)+(RTstd*std(RT_FR))))); % discard extreme outlier trials based on STD
    ACC_FR = ACC_FR(find(RT_FR<RTthresh)); RT_FR = RT_FR(find(RT_FR<RTthresh)); % discard extreme outlier trials based on absolute value
    
    if p==1
        ALL_RTc = RT_FR(find(ACC_FR==1));
        ALL_RTe = RT_FR(find(ACC_FR==0));
    else
        ALL_RTc(end+1:end+length(find(ACC_FR==1))) = RT_FR(find(ACC_FR==1));
        ALL_RTe(end+1:end+length(find(ACC_FR==0))) = RT_FR(find(ACC_FR==0));
    end
    
end

% Accuracy
n_trials = length(ALL_RTc)+length(ALL_RTe);
Oacc = length(ALL_RTc)/n_trials;

% Calculating 10 observed RT quantiles and trial frequencies therein
Qc = quantile(ALL_RTc,(10:10:90)./100);   % getting correct RT quantiles
Oc = ones(1,10).*(Oacc*n_trials*(10/100));

Qe = quantile(ALL_RTe,(10:10:90)./100);   % getting error RT quantiles
Oe = ones(1,10).*((1-Oacc)*n_trials*(10/100));


%%%%%%%%%%%%%%%%%%%%
%%% PSO settings %%%
%%%%%%%%%%%%%%%%%%%%

% Defining PSO options (see pso_Trelea_vectorized.m for details)
  P(1)=0;  P(2)=1000;     P(3)=70;    P(4:13)=[1.6 1.9 0.9 0.4 400 1e-25 250 NaN 0 1];
% display  n_iterations  n_particles       acceleration, inertia, tolerance, etc

% Seeding first n particles with parameters drawn from realistic distributions
n_seeded = 25;
PSOseedValue=[];

PSOseedValue(1:n_seeded,end+1) = seeds.lamda(1)+(randn(n_seeded,1).*seeds.lamda(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.lamda(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.lamda(1)),end) = range_p.lamda(1); end % just in case there are any too-low values
if ~isempty(find(PSOseedValue(:,end)>range_p.lamda(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.lamda(2)),end) = range_p.lamda(2); end % just in case there are any too-high values

PSOseedValue(1:n_seeded,end+1) = seeds.v(1)+(randn(n_seeded,1).*seeds.v(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.v(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.v(1)),end) = range_p.v(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.v(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.v(2)),end) = range_p.v(2); end

PSOseedValue(1:n_seeded,end+1) = seeds.theta(1)+(randn(n_seeded,1).*seeds.theta(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.theta(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.theta(1)),end) = range_p.theta(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.theta(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.theta(2)),end) = range_p.theta(2); end

PSOseedValue(1:n_seeded,end+1) = seeds.noise(1)+(randn(n_seeded,1).*seeds.noise(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.noise(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.noise(1)),end) = range_p.noise(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.noise(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.noise(2)),end) = range_p.noise(2); end

PSOseedValue(1:n_seeded,end+1) = seeds.eta(1)+(randn(n_seeded,1).*seeds.eta(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.eta(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.eta(1)),end) = range_p.eta(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.eta(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.eta(2)),end) = range_p.eta(2); end

PSOseedValue(1:n_seeded,end+1) = seeds.glamda(1)+(randn(n_seeded,1).*seeds.glamda(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.glamda(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.glamda(1)),end) = range_p.glamda(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.glamda(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.glamda(2)),end) = range_p.glamda(2); end

PSOseedValue(1:n_seeded,end+1) = seeds.gk(1)+(randn(n_seeded,1).*seeds.gk(2));
if ~isempty(find(PSOseedValue(:,end)<range_p.gk(1))), PSOseedValue(find(PSOseedValue(:,end)<range_p.gk(1)),end) = range_p.gk(1); end
if ~isempty(find(PSOseedValue(:,end)>range_p.gk(2))), PSOseedValue(find(PSOseedValue(:,end)>range_p.gk(2)),end) = range_p.gk(2); end

% Concatenating parameter ranges
par_range = [range_p.lamda; range_p.v; range_p.theta; range_p.noise; range_p.eta; range_p.glamda; range_p.gk];

% Randomly seeding remaining particles within prespecified bounds
PSOseedValue(size(PSOseedValue,1)+1:P(3),1:size(PSOseedValue,2)) = normmat(rand([P(3)-n_seeded,size(PSOseedValue,2)]),par_range',1);

% Constraining noise to always be less than stimulus input (otherwise fits tend to yield noise-driven simulated decision variables)
for s = 1:size(PSOseedValue,1)
    if PSOseedValue(s,4)>=PSOseedValue(s,2)
        PSOseedValue(s,4) =  range_p.noise(1)+(PSOseedValue(s,2)-range_p.noise(1)).*rand(1); % redrawing noise bounded by original min and this iteration's v
    end
end

% Running PSO routine
[output,tr,te] = pso_Trelea_vectorized_LCA('ML_from_LCA_FR',length(mv),mv,par_range,0,P,'goplotpso',PSOseedValue);

% Saving output
pm_best = output(1:end-1);
L = output(end);

AIC = (2*L)+(2*length(pm_best));
BIC = (2*L)+(length(pm_best)*log(n_trials));

save([savepath,'LCA_FR_fit.mat'],...
    'pm_best','L','AIC','BIC','tr','te','ALL_RTc','ALL_RTe','n_trials','Oacc','Qc','Qe','Oc','Oe')




