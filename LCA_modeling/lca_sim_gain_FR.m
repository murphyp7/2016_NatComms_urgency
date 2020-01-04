% SIMULATE BEHAVIOUR FROM LEAKY COMPETING ACCUMULATOR MODEL %%%%%%

function [RT,ACC]=lca_sim_gain_FR(Pars,nsims)

% SETTING PARAMETERS
lamda=Pars(1);      % leak
alpha=Pars(11);      % recurent exhitation
beta=Pars(12);       % inhibition
i0=Pars(8);         % baseline input
v=Pars(2);          % input to accumulator 1 (input to accumulator 2 is constrained to be 1-v, such that i1+i2=1)
theta=Pars(3);      % decision bound
proc_noise=Pars(4); % processing noise
eta=Pars(5);        % eta (drift rate variability)
Ter=Pars(10);        % non-decision time

u0 = Pars(9);      % baseline gain
lambda = Pars(6);  % time to peak
k = Pars(7);       % shape

dt=0.01;      % time step for simulations (in seconds)
prestimt=1;   % pre-stimulus time for just baseline input, allowing network to settle into stable state prior to stimulus onset

maxt=5.6;     % max time to simulate (in seconds)

N=nsims; % number of trials to simulate

[g,~,~] = urgency_f(u0,lambda,k,maxt-prestimt,dt,'up');  % gain time-course
g = [ones(1,prestimt/dt).*u0 g];  % adding baseline gain values to pre-stimulus period
g(end) = []; % getting rid of final extra g value, which script will estimate unnecessarily

% Drawing trial-by-trial drift rates
if eta>0  
    v1s = v+(eta.*randn(N,1));
    v1s(v1s>1)=1; v1s(v1s<0)=0;
    v2s = ones(N,1)-v1s;
else
    v1s = v.*ones(N,1);
    v2s = (1-v).*ones(N,1);
end

% Drawing trial-by-trial noise time-series (pre-specifying here speeds up simulations a lot)
noise=zeros(2,N,maxt/dt);
noise1=randn(N,size(noise,3)).*proc_noise; % generating noise for correct accumulator
noise2=randn(N,size(noise,3)).*proc_noise; % generating noise for incorrect accumulator

% Drawing trial-by-trial noiseless input time-series given each trial's drift rate
input=zeros(2,N,maxt/dt);
input1=repmat((v1s+i0),1,maxt/dt); % generating stimulus input for correct accumulator
input2=repmat((v2s+i0),1,maxt/dt); % generating stimulus input for incorrect accumulator

% Create single, noiseless pre-stim time course used for all simulations
prestim_in = gain_f_lca_DecNoise_fast(i0,g(1),theta);  % estimating baseline gain-transformed input
preP = zeros(prestimt/dt,2);
for t = 2:(prestimt/dt);
    x1t = gain_f_lca_DecNoise_fast(preP(t-1,1),g(t),theta);
    x2t = gain_f_lca_DecNoise_fast(preP(t-1,2),g(t),theta);
    preP(t,1) = max(0,(preP(t-1,1)*(1-lamda))+(alpha*x1t)-(beta*x2t)+prestim_in);
    preP(t,2) = max(0,(preP(t-1,2)*(1-lamda))+(alpha*x2t)-(beta*x1t)+prestim_in);
end

% Passing input and noise time-series through transfer function - speeds up a lot by taking advantage of fixed gain for given time step
for t = ((prestimt/dt)+1):(maxt/dt);
    g_in = [noise1(:,t); noise2(:,t); input1(:,t); input2(:,t)];  % generating single vector of input to pass through transfer function
    cutoffs = [0 N N*2 N*3 N*4];
    g_out = gain_f_lca_DecNoise_fast(g_in,g(t),theta);
    noise1(:,t) = g_out(cutoffs(1)+1:cutoffs(2));
    noise2(:,t) = g_out(cutoffs(2)+1:cutoffs(3));
    input1(:,t) = g_out(cutoffs(3)+1:cutoffs(4));
    input2(:,t) = g_out(cutoffs(4)+1:cutoffs(5));
end

% Looping through samples after stimulus onset
P1 = repmat(preP(:,1),1,N)';
P2 = repmat(preP(:,2),1,N)';
for t = ((prestimt/dt)+1):(maxt/dt);
    x = gain_f_lca_DecNoise_fast([P1(:,t-1); P2(:,t-1)],g(t),theta);  % passing previous activations through transfer function
    
    % Update      act minus leak      rec ex            lat inh         new input    noise
    P1(:,t) = (P1(:,t-1).*(1-lamda))+(alpha.*x(1:N))-(beta.*x(N+1:end))+input1(:,t)+noise1(:,t);
    P2(:,t) = (P2(:,t-1).*(1-lamda))+(alpha.*x(N+1:end))-(beta.*x(1:N))+input2(:,t)+noise2(:,t);
    
    % Imposing lower bound on activations (replace any negative values with zeros)
    P1(P1(:,t)<0,t) = 0;
    P2(P2(:,t)<0,t) = 0;
end

% Classifying accuracy & pulling RTs
ACC=nan(N,1); RT=nan(N,1); % prespecifying vector of accuracy markers
[I1,J1] = find(P1>=theta);  % getting latencies of all passage times for each accumulator/trial
[I2,J2] = find(P2>=theta);

[I1,Iindex] = unique(I1,'first'); J1 = J1(Iindex);  % keeping only *first* passage times for each accumulator/trial
[I2,Iindex] = unique(I2,'first'); J2 = J2(Iindex);

J1 = ((J1-(prestimt/dt)).*dt)+Ter;  % convert first passage times to RTs
RT1 = nan(N,1); RT1(I1) = J1;  % and plug into vector with length = nsims

J2 = ((J2-(prestimt/dt)).*dt)+Ter;
RT2 = nan(N,1); RT2(I2) = J2;

ts = (1:N)';  % vector of trial numbers for later indexing
dual_pass = find(ismember(ts,I1) & ismember(ts,I2));  % vector of trial numbers with dual bound crossings for later indexing

ACC((ismember(ts,I1) & RT1>=Ter) & (~ismember(ts,I2) | ismember(ts,dual_pass(RT1(dual_pass)-RT2(dual_pass)<=0)))) = 1;  % correct
ACC((ismember(ts,I2) & RT2>=Ter) & (~ismember(ts,I1) | ismember(ts,dual_pass(RT2(dual_pass)-RT1(dual_pass)<=0)))) = 0;  % correct

RT(ACC==1) = RT1(ACC==1);
RT(ACC==0) = RT2(ACC==0);
