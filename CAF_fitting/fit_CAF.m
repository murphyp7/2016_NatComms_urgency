% Script for running piece-wise logistic regression fit to conditional
% accuracy data at the single-trial level
%
% First slope (accounting for fast errors) will be constrained to >= 0
%
% Peter Murphy, UKE Hamburg, 04/01/2020

clc
clear, close all

tic

% Paths
funpath = 'path to functions';
loadpath = 'path to data';
figpath = 'path to save figure';

addpath(genpath(funpath))

% File IDs
allsubj = {'109', '110', '112', '113', '114', '116', '117', '119', '120', '121', '124', '125', '126', '127', '128', '129', '131', '132', '134', '135', '136', '137', '138', '139', '140', '142'};
n_blocks = 5;

% CAF fitting settings
max_inflect = 1;  % maximum latency of inflection point in piece-wise fit (in seconds)
min_trials = 15;  % minimum number of trials for fit left of inflection point
stepsize = 0.01;  % size of exhuastive search increments between consecutive inflection points (in seconds)
options = optimset('MaxIter',5000,'MaxFunEvals',5000,'TolFun',1e-5,'TolX',1e-5);  % set Simplex options to perform rigorous search
real_ts_DL = 0:0.05:1.5;  % time points at which to plot fitted CAFs

% Observed CAF settings
n_bins = 25;  % number bins per participant for CAF plots
min_bin = 10;  % minimum number of trials in final bin - if below this number, last 2 bins will be merged

% Looping through subjects/blocks
for p = 1:length(allsubj);
    
    disp(sprintf('Processing subject %d of %d...',p,length(allsubj)))
    
    subj = allsubj{p};
    
    for b = 1:n_blocks
        % Extracting accuracy & RTs
        load([loadpath,subj,'_',num2str(b),'.mat'])
        if b==1
            ACC_DL = data_block(find(data_block(:,4)>0),2);
            RT_DL = data_block(find(data_block(:,4)>0),4);
            All_RT_DL = data_block(:,4);
            Blockwise_RT_DL = data_block(:,4);
        else ACC_DL(end+1:end+length(find(data_block(:,4)>0))) = data_block(find(data_block(:,4)>0),2);
            RT_DL(end+1:end+length(find(data_block(:,4)>0))) = data_block(find(data_block(:,4)>0),4);
            All_RT_DL(end+1:end+length(data_block(:,4))) = data_block(:,4);
            Blockwise_RT_DL(:,b) = data_block(:,4);
        end
    end
    
    % Collating all RTs per condition/accuracy-type across subjects
    if p==1
        ALL_RTc_DL = RT_DL(find(ACC_DL==1));
        ALL_RTe_DL = RT_DL(find(ACC_DL==0));
    else
        ALL_RTc_DL(end+1:end+length(find(ACC_DL==1))) = RT_DL(find(ACC_DL==1));
        ALL_RTe_DL(end+1:end+length(find(ACC_DL==0))) = RT_DL(find(ACC_DL==0));
    end
    
    % Calculating average accuracy/RT/p(miss)
    GA_ACC_DL(p,1) = sum(ACC_DL)/length(ACC_DL)*100;
    GA_RTc_DL(p,1) = median(RT_DL(find(ACC_DL==1)));
    GA_RTe_DL(p,1) = median(RT_DL(find(ACC_DL==0)));
    GA_miss_DL(p,1) = length(All_RT_DL(find(All_RT_DL==0)))/length(All_RT_DL)*100;
    
    %%%%%% observed CAF %%%%%%
    % Sorting trials by RT
    [RT_DLs,sorting] = sort(RT_DL);
    ACC_DLs = ACC_DL(sorting); 
    % Constructing bin edges & merging last 2 bins if final bin has low trial count
    breaks = 1:length(RT_DL)/n_bins:length(RT_DL);
    if breaks(end)~=length(RT_DL), breaks(end+1)=length(RT_DL); end
    if breaks(end)-breaks(end-1)<min_bin, breaks(end-1)=[]; end    
    % Pulling out measures from each bin and storing
    RT_DL_bin=[]; ACC_DL_bin=[];
    for bin = 1:length(breaks)-1;
        RT_DL_bin(length(RT_DL_bin)+1,1) = mean(RT_DLs(ceil(breaks(bin)):floor(breaks(bin+1))));
        ACC_DL_bin(length(ACC_DL_bin)+1,1) = mean(ACC_DLs(ceil(breaks(bin)):floor(breaks(bin+1))));
    end
    GA_RT_DL_bin(1:length(RT_DL_bin),p) = RT_DL_bin;
    GA_ACC_DL_bin(1:length(ACC_DL_bin),p) = ACC_DL_bin;    
    
    %%%%%% fit %%%%%%
    % Getting latency at which to start testing inflection points
    min_lat = (ceil(RT_DLs(min_trials).*100))./100;
    max_lat = (ceil(RT_DLs(end-min_trials-1).*100))./100;
    % Vector of inflection points to test
    if max_lat>max_inflect
        breaks = min_lat:stepsize:max_inflect;
    else breaks = min_lat:stepsize:max_lat;  % contingency in case there aren't enough slow trials for testing later inflection points
    end
    % Start exhuastive search of inflection points
    lParams=[]; rParams=[]; lSSR=[]; rSSR=[]; all_inf=[];
    for i = 1:length(breaks)
        % Pull current inflection point and centre RTs around this
        c_inf = breaks(i);
        c_RT = RT_DLs-c_inf;
        % Fit logistic model to LEFT, THEN RIGHT of inflection point & save params + ssr
        data = [c_RT(find(c_RT<=0)) ACC_DLs(find(c_RT<=0))];
        [lParams(end+1,:),lSSR(end+1,1)] = fminsearchbnd(@(params) logistic_fit(params,data),[1.5 1],[-inf 0],[+inf +inf],options);  % running minimisation routine
        data = [c_RT(find(c_RT>0)) ACC_DLs(find(c_RT>0))];
        [rParams(end+1,:),rSSR(end+1,1)] = fminsearchbnd(@(params) logistic_fit(params,data),[lParams(end,1) -0.5],[lParams(end,1) -inf],[lParams(end,1) 0],options);  % running minimisation routine
        all_inf(end+1,1) = c_inf;
        % Fit logistic model to RIGHT, THEN LEFT of inflection point & save params + ssr
        data = [c_RT(find(c_RT>0)) ACC_DLs(find(c_RT>0))];
        [rParams(end+1,:),rSSR(end+1,1)] = fminsearchbnd(@(params) logistic_fit(params,data),[1.5 -0.5],[-inf -inf],[+inf 0],options);  % running minimisation routine
        data = [c_RT(find(c_RT<=0)) ACC_DLs(find(c_RT<=0))];
        [lParams(end+1,:),lSSR(end+1,1)] = fminsearchbnd(@(params) logistic_fit(params,data),[rParams(end,1) 1],[rParams(end,1) 0],[rParams(end,1) +inf],options);  % running minimisation routine
        all_inf(end+1,1) = c_inf;
    end
    % Pick inflection point and parameters with minimum combined ssr
    best_inf = find((lSSR+rSSR)==min(lSSR+rSSR));
    DL_inflect(p,1) = all_inf(best_inf);
    DL_lParams(p,:) = lParams(best_inf,:);
    DL_rParams(p,:) = rParams(best_inf,:);
    % Creating model-predicted CAF
    CAFpred = [];
    ts_DL = -(DL_inflect(p)-real_ts_DL(1)):(real_ts_DL(2)-real_ts_DL(1)):(real_ts_DL(end)-DL_inflect(p));
    for t = 1:length(ts_DL);
        if ts_DL(t) <= 0
            CAFpred(t,1) = 1./(1+exp(-(DL_lParams(p,1)+(DL_lParams(p,2)*ts_DL(t)))));
        else
            CAFpred(t,1) = 1./(1+exp(-(DL_rParams(p,1)+(DL_rParams(p,2)*ts_DL(t)))));
        end
    end
    GA_CAFpred_DL(:,p) = CAFpred;
    % Calculating model-estimated time at which performance is at chance
    GA_chance_time_DL(p,1) = (DL_rParams(p,1)/-DL_rParams(p,2))+DL_inflect(p);
    % Calculating model-estimated accuracy at deadline
    GA_ACC_at_DL(p,1) = CAFpred(end);
        
end

% Plotting grand-average observed & fitted CAFs
include = ones(size(allsubj));

CAFpred_DL_err = std(GA_CAFpred_DL(:,find(include==1)),[],2)./sqrt(size(GA_CAFpred_DL(:,find(include==1)),2));

RTbin_DL_err = std(GA_RT_DL_bin(:,find(include==1)),[],2)./sqrt(size(GA_RT_DL_bin(:,find(include==1)),2));
mRTbin_DL = mean(GA_RT_DL_bin(:,find(include==1)),2);

ACCbin_DL_err = std(GA_ACC_DL_bin(:,find(include==1)),[],2)./sqrt(size(GA_ACC_DL_bin(:,find(include==1)),2));
mACCbin_DL = mean(GA_ACC_DL_bin(:,find(include==1)),2);

figure, hold on
shadedErrorBar(real_ts_DL,mean(GA_CAFpred_DL(:,find(include==1)),2),CAFpred_DL_err,{'Color',[0 0 0],'LineWidth',2},1),
for b=1:length(RTbin_DL_err)
    L1=line([mRTbin_DL(b) mRTbin_DL(b)],[mACCbin_DL(b)-ACCbin_DL_err mACCbin_DL(b)+ACCbin_DL_err]); set(L1,'LineWidth',1,'Color',[0.7 0.7 0.7]),
    L2=line([mRTbin_DL(b)-RTbin_DL_err mRTbin_DL(b)+RTbin_DL_err],[mACCbin_DL(b) mACCbin_DL(b)]); set(L2,'LineWidth',1,'Color',[0.7 0.7 0.7]),
end
S=scatter(mRTbin_DL,mACCbin_DL,70,[1 0 0]); set(S,'LineWidth',1.5, 'MarkerFaceColor',[1 1 1], 'MarkerEdgeColor',[0.2 0.2 0.2])
xlabel('RT (s)'); ylabel('P_c_o_r_r_e_c_t'); xlim([0.3 max(real_ts_DL)+0.1]), ylim([0.45 1])
hold off

save([figpath,'CAF_bin_and_fit.mat'],'real_ts_DL','GA_CAFpred_DL','GA_RT_DL_bin','GA_ACC_DL_bin','include')

% Plotting per-subject observed & fitted CAFs
figure
for p = 1:10;
    subplot(5,2,p)
    hold on
    S1=scatter(GA_RT_DL_bin(:,p),GA_ACC_DL_bin(:,p),70,[0 0 0]); set(S1,'LineWidth',1.5, 'MarkerFaceColor',[1 1 1], 'MarkerEdgeColor',[0.2 0.2 0.2])
    L=line([DL_inflect(p,1),DL_inflect(p,1)],[0.4 1]);  set(L,'LineStyle','--','LineWidth',1.5,'Color',[0 0 0]),
    plot(real_ts_DL,GA_CAFpred_DL(:,p),'Color',[0 0 0],'LineWidth',3)
    xlabel('RT (s)'); ylabel('P_c_o_r_r_e_c_t'); xlim([0.2 max(real_ts_DL)+0.1]), ylim([0.4 1]), title(allsubj{p})
    hold off
end

figure
for p = 11:20;
    subplot(5,2,p-10)
    hold on
    S1=scatter(GA_RT_DL_bin(:,p),GA_ACC_DL_bin(:,p),70,[1 0 0]); set(S1,'LineWidth',1.5, 'MarkerFaceColor',[1 1 1], 'MarkerEdgeColor',[0.2 0.2 0.2])
    L=line([DL_inflect(p,1),DL_inflect(p,1)],[0.4 1]);  set(L,'LineStyle','--','LineWidth',1.5,'Color',[0 0 0]),
    plot(real_ts_DL,GA_CAFpred_DL(:,p),'Color',[0 0 0],'LineWidth',3)
    xlabel('RT (s)'); ylabel('P_c_o_r_r_e_c_t'); xlim([0.2 max(real_ts_DL)+0.1]), ylim([0.4 1]), title(allsubj{p})
    hold off
end

figure
for p = 21:26;
    subplot(5,2,p-20)
    hold on
    S1=scatter(GA_RT_DL_bin(:,p),GA_ACC_DL_bin(:,p),70,[1 0 0]); set(S1,'LineWidth',1.5, 'MarkerFaceColor',[1 1 1], 'MarkerEdgeColor',[0.2 0.2 0.2])
    L=line([DL_inflect(p,1),DL_inflect(p,1)],[0.4 1]);  set(L,'LineStyle','--','LineWidth',1.5,'Color',[0 0 0]),
    plot(real_ts_DL,GA_CAFpred_DL(:,p),'Color',[0 0 0],'LineWidth',3)
    xlabel('RT (s)'); ylabel('P_c_o_r_r_e_c_t'); xlim([0.2 max(real_ts_DL)+0.1]), ylim([0.4 1]), title(allsubj{p})
    hold off
end

% T-tests on important quantities
[~,p3] = ttest(GA_chance_time_DL(find(include==1)),1.5);  % testing whether DL performance reaches chance levels prior to deadline
[~,p4] = ttest(GA_ACC_at_DL(find(include==1)),0.5);  % testing whether accuracy @ DL is different from chance

disp(sprintf('\nDL latency of chance accuracy:'))
disp(sprintf('Time = %2.3f s.....p against 1.5 s deadline = %2.3f',mean(GA_chance_time_DL(find(include==1))),p3))
disp(sprintf('\nDL accuracy @ deadline:'))
disp(sprintf('P(correct) = %2.3f.....p against 0.5 = %2.3f',mean(GA_ACC_at_DL(find(include==1))),p4))
disp(sprintf('\nAverage proportion of misses:'))
disp(sprintf('Percentage = %2.3f, SD = %2.3f',mean(GA_miss_DL(find(include==1))),std(GA_miss_DL(find(include==1)))))



