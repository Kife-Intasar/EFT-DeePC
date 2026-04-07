function suite = run_ablation(isQuick)


if nargin < 1, isQuick = false; end
if isQuick, R = 1; else, R = 5; end

padFrac  = 0.10;
saveFile = 'ablation_suite_ppsn_lit_vr.mat';

base = deepc_default_cfg();
base.P = 50; base.G = 20; 
seed_data_fixed = base.seed_data; 
base.print.verbose       = false;
base.print.best_each_gen = false;

% Ensure this field exists globally (prevents "window_max_cols" error)
if ~isfield(base,'online'), base.online = struct(); end
if ~isfield(base.online,'window_max_cols'), base.online.window_max_cols = inf; end

% Budget info
evalBudget = base.P*(base.G+1);
fprintf('Eval budget per run: %d candidates (P*(G+1))\n', evalBudget);

cfgs_base  = {};
names_base = {};

%% =========================
% YOUR METHOD (main)
%% =========================
c = base;
c.name = 'METHOD_multi_CVaR_median_RO_gated';
cfgs_base{end+1}  = c;  names_base{end+1} = c.name;

%% =========================
% EXPECTED ABLATIONS
%% =========================
c = base; c.obj.use_cvar = false;
c.name = 'ABL_NoCVaR';
cfgs_base{end+1} = c; names_base{end+1} = c.name;

c = base; c.ro.enable = false;
c.name = 'ABL_NoRO_fullBasis';
cfgs_base{end+1} = c; names_base{end+1} = c.name;

c = base; c.online.update_mode = 'never';
c.name = 'ABL_NoOnlineUpdate';
cfgs_base{end+1} = c; names_base{end+1} = c.name;

c = base;
c.obj.use_robust = false; c.obj.S = 1;
c.obj.use_cvar = false; c.obj.time_stat='mean';
c.name = 'ABL_SingleScenario_S1';
cfgs_base{end+1} = c; names_base{end+1} = c.name;


c = base; c.obj.time_stat = 'mean';
c.name = 'ABL_TimeMean';
cfgs_base{end+1} = c; names_base{end+1} = c.name;

c = base;
c.name = 'ABL_FixedHorizon_tuneRegs';
Tini_fix = 35; N_fix = 45; Delta_fix = 200;
c.lb(1:3) = [Tini_fix N_fix Delta_fix];
c.ub(1:3) = [Tini_fix N_fix Delta_fix];
cfgs_base{end+1} = c; names_base{end+1} = c.name;

c = base;
c.name = 'ABL_OptimizedHorizon';
cfgs_base{end+1} = c; names_base{end+1} = c.name;

%% =========================
% Expand to MOEA/D + NSGA-II (random stays single)
%% =========================
algos = {'moead','nsga2'};

cfgs  = {};
names = {};
baseId = [];   % base config index (for fair seeding across algos)

for b = 1:numel(cfgs_base)
    cb = cfgs_base{b};

    % Ensure field exists on every config (robust against older cfgs)
    if ~isfield(cb,'online'), cb.online = struct(); end
    if ~isfield(cb.online,'window_max_cols'), cb.online.window_max_cols = inf; end

    if isfield(cb,'algo') && strcmpi(cb.algo,'random')
        cfgs{end+1}  = cb;                     %#ok<AGROW>
        names{end+1} = cb.name;                %#ok<AGROW>
        baseId(end+1)= b;                      %#ok<AGROW>
    else
        for a = 1:numel(algos)
            c = cb;
            c.algo = algos{a};

            cfgs{end+1}  = c;                                  %#ok<AGROW>
            names{end+1} = sprintf('%s_%s', cb.name, upper(c.algo)); %#ok<AGROW>
            baseId(end+1)= b;                                  %#ok<AGROW>
        end
    end
end

suite = struct();
suite.names = names;
suite.cfgs  = cfgs;
suite.R     = R;
suite.runs  = cell(numel(cfgs), R);

fprintf('Running suite: %d configs x %d repeats = %d runs\n', numel(cfgs), R, numel(cfgs)*R);

%% =========================
% Run
%% =========================
for i = 1:numel(cfgs)
    for r = 1:R
        cfg = cfgs{i};

        % FAIR SEEDING:
        % - Same offline dataset for same base config + repeat (independent of algo)
        % - Different algo seed to vary search
        % % seed_core = 10000*baseId(i) + 100*r;
        % % 
        % % cfg.seed_data = seed_core;
        % % cfg.seed_algo = seed_core + sum(double(upper(cfg.algo)));

        seed_core = seed_data_fixed + 100*r;

        cfg.seed_data = seed_core;
        cfg.seed_algo = seed_core + 10000*i + sum(double(upper(cfg.algo)));

        % Ensure again in case cfg was older
        if ~isfield(cfg,'online'), cfg.online = struct(); end
        if ~isfield(cfg.online,'window_max_cols'), cfg.online.window_max_cols = inf; end

        fprintf('[%2d/%2d] %s | rep %d/%d | algo=%s | evalBudget=%d | seed_data=%d seed_algo=%d\n', ...
            i, numel(cfgs), names{i}, r, R, cfg.algo, cfg.P*(cfg.G+1), cfg.seed_data, cfg.seed_algo);

        suite.runs{i,r} = deepc_run(cfg);
    end
end

save(saveFile,'suite');
fprintf('Saved: %s\n', saveFile);

T = summarize_suite(suite, padFrac);
disp(T);
writetable(T,'ablation_summary_ppsn_lit_vr.csv');
fprintf('Wrote: ablation_summary_ppsn_lit_vr.csv\n');
end
