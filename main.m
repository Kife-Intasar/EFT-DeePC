
clear all; clc;

% ------------------------------------------------------------
% 0) Base config + fairness seeds (same offline data everywhere)
% ------------------------------------------------------------
base = deepc_default_cfg();
base.bounds.mode = 'pilot_data_driven'; % 'manual'| 'data_driven' | 'pilot_data_driven'
base.prob.Ksim  = 2100; 
base.prob.model = 'rollover';     
base.prob.Ts    = 0.1;     
%base.prob.ref_val = 0;            % rollover + LFC: regulate to 0 is sensible

seed_data_fixed = 111;             % SAME offline data for all methods/algos
repeat_id = 1;                     % change if you want multiple repeats
seed_core = seed_data_fixed + 100*repeat_id;

base.seed_data = seed_core;

b1 = cfg_baseline_coulson19_lit(base);   
b2 = cfg_baseline_teutsch23_lit(base);  
b3 = cfg_baseline_shi23_lit(base);      
b4 = cfg_baseline_fixedpaper(base); 
b5 = cfg_baseline_worstcase(base); 

% Ensure window_max_cols exists everywhere (prevents your error)
if ~isfield(base,'online'), base.online = struct(); end
if ~isfield(base.online,'window_max_cols'), base.online.window_max_cols = inf; end
if ~isfield(b1,'online'), b1.online = struct(); end
if ~isfield(b1.online,'window_max_cols'), b1.online.window_max_cols = inf; end
if ~isfield(b2,'online'), b2.online = struct(); end
if ~isfield(b2.online,'window_max_cols'), b2.online.window_max_cols = inf; end
if ~isfield(b3,'online'), b3.online = struct(); end
if ~isfield(b3.online,'window_max_cols'), b3.online.window_max_cols = inf; end

% ------------------------------------------------------------
% 2) Run MOEA/D: method + 3 baselines
% ------------------------------------------------------------
cfgM = base; cfgM.algo = 'moead';
cfgM.seed_algo = seed_core + sum(double('METHOD_MOEAD'));  
outM = deepc_run(cfgM);

% ------------------------------------------------------------
% 3) Run NSGA-II: method + 3 baselines
% ------------------------------------------------------------
cfgN = base; cfgN.algo = 'nsga2';
cfgN.seed_algo = seed_core + sum(double('METHOD_NSGA2'));
outN = deepc_run(cfgN);



% ------------------------------------------------------------
% 3) Run Random: method + 3 baselines
% ------------------------------------------------------------
cfgR = base; cfgR.algo = 'random';
cfgR.seed_algo = seed_core + sum(double('METHOD_RANDOM'));
outR = deepc_run(cfgR);




% ------------------------------------------------------------
% 4) Call your comparison paper-figure function (METHOD RO/Full + 3 baselines)
% ------------------------------------------------------------
whichBest = 'ctrl';   % 'ctrl' or 'fast'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Storing the Tuned Variables #######
% Fix names first so lengths match
outB_M_P  = {outM};
nameB_M_P = {'Proposed Method'};

n = numel(outB_M_P);
assert(n == numel(nameB_M_P), 'Number of structs and names must match.');

ctrl_x = cell(n,1);
time_x = cell(n,1);

for i = 1:n
    ctrl_x{i} = outB_M_P{i}.best.ctrl_x;
    time_x{i} = outB_M_P{i}.best.time_x;
end

T = table(nameB_M_P(:), ctrl_x, time_x, ...
    'VariableNames', {'Name', 'ctrl_x', 'time_x'});

writetable(T, 'best_fields_M.xlsx', 'Sheet', 'best_fields');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Storing the Tuned Variables #######
% Fix names first so lengths match
outB_N_P  = { outN};
nameB_N_P = {'Proposed Method'};

n = numel(outB_N_P);
assert(n == numel(nameB_N_P), 'Number of structs and names must match.');

ctrl_x = cell(n,1);
time_x = cell(n,1);

for i = 1:n
    ctrl_x{i} = outB_N_P{i}.best.ctrl_x;
    time_x{i} = outB_N_P{i}.best.time_x;
end

T = table(nameB_N_P(:), ctrl_x, time_x, ...
    'VariableNames', {'Name', 'ctrl_x', 'time_x'});

writetable(T, 'best_fields_N.xlsx', 'Sheet', 'best_fields');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% Ablation %%%%%%%%%%%%%%%%%%%%

suite2 = run_ablation_suite_ppsn_lit(false);

