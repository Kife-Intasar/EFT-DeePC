function cfg = deepc_default_cfg()

cfg = struct();

%% ------------------------
% Reproducibility (FAIRNESS)
%% ------------------------
cfg.seed      = 1;      % legacy (kept)
cfg.seed_data = 111;    
cfg.seed_algo = 222;    

%% ------------------------
% Algorithm / budget
%% ------------------------
cfg.algo = 'moead';      % 'moead' | 'nsga2' | 'random'
cfg.P =50;
cfg.G = 20;              % total evals = P*(G+1)


%% ------------------------
% Data-Driven Bound mode
%% ------------------------

% Decision x = [Tini, N, Delta, log10(lam_g), log10(lam_sig), log10(sig_thr)]
cfg.lb = [10, 10,  200,  -3,  -6,  -7];
cfg.ub = [35, 45,  600,    1,   2,   -2];
cfg.super_lb = [5, 5,  100,  -8,  -8,  -10];
cfg.super_ub = [50, 60,  900,    4,   8,   -1];


cfg.bounds = struct();
cfg.bounds.mode = 'manual'; % 'pilot_data_driven' | 'data-driven'
cfg.bounds.nPilot = 300;
cfg.bounds.keepFrac = 0.20;
cfg.bounds.padFrac = 0.10; 
cfg.bounds.minKeep = 20; 
cfg.bounds.lb_manual = cfg.lb;
cfg.bounds.ub_manual = cfg.ub; 

%% ------------------------
% NSGA-II params
%% ------------------------
cfg.nsga2.pc = 0.9;
cfg.nsga2.pm = 0.3;
cfg.nsga2.eta_c = 15;
cfg.nsga2.eta_m = 20;

%% ------------------------
% MOEA/D params
%% ------------------------
cfg.moead.Tn = 20;
cfg.moead.F  = 0.5;
cfg.moead.CR = 0.9;
cfg.moead.pm = 0.2;
cfg.moead.eta_m = 20;

%% ------------------------
% Objective settings (robust + tail-aware)
%% ------------------------
cfg.obj.use_robust = true;
cfg.obj.S = 10; 
cfg.obj.use_cvar = true;
cfg.obj.alpha_cvar = 1.0;

cfg.obj.ctrl_stat = 'mean';     % 'mean' or 'max'
cfg.obj.time_stat = 'median';   % 'median' | 'mean' | 'max'
cfg.obj.gamma_time = 1.0;

cfg.obj.tailFrac = 0.20;
tailCount = max(2, ceil(cfg.obj.tailFrac * cfg.obj.S));
tailCount = min(tailCount, cfg.obj.S);
cfg.obj.q = (cfg.obj.S - tailCount + 1) / cfg.obj.S;
cfg.obj.q = min(max(cfg.obj.q, 0.5), 0.99);

%% ------------------------
% Scenario variability
%% ------------------------
cfg.scen.x0_sigma = 0.05;
cfg.scen.lambda_phase_jitter = true;
cfg.scen.lambda_shift_max = 400;
cfg.scen.seed_offset = 0;   

%% ------------------------
% Reduced-order + online update
%% ------------------------
cfg.ro.enable = true;
cfg.ro.ra_min_floor = false; %true
cfg.ro.ra_min = [];      % default: uses m*Tini + n
% cfg.ro.ra_min = 30;    % if you want it very small for “fast”

cfg.online.update_mode = 'gated'; % 'gated' | 'always' | 'never'
cfg.online.update_period = 15;
cfg.online.window_max_cols = 600;

%% ------------------------
% Plant/model selection + references
%% ------------------------
cfg.prob.model = 'ltv';      % 'ltv' | 'rollover' | 'quadruple_tank' | 'lfc'
cfg.prob.Ts = [];            % optional override for some models

% Use larger Ksim while tuning, otherwise optimizer may pick "lazy" solutions
cfg.prob.Ksim = 2100;         
cfg.prob.Tfull = 3000;       

% Reference handling (IMPORTANT):
cfg.prob.auto_ref = true;    
cfg.prob.ref_scale = 0.85;   
cfg.prob.ref_val = [];       
cfg.prob.ref_vec = [];       
cfg.prob.ref_fun = [];       

%% ------------------------
% Constraints (feasibility)
%% ------------------------
cfg.con.kappa_max = 1e7;
cfg.con.sigma_min = 1e-10;
cfg.con.fail_rate_max = 0.05;
cfg.con.x_max = 30;
cfg.con.e_tail_frac = 0.20;
cfg.con.e_tail_min_steps = 10;
cfg.con.use_tail_constraint = true;   
cfg.con.e_tail_max = NaN; 
cfg.con.e_tail_stat = 'mean';    % 'max' | 'mean' | 'cvar'

%% ------------------------
% Numerical safety (KKT regularization)
%% ------------------------
cfg.con.lam_g_min   = 1e-6; %1e-3;
cfg.con.lam_sig_min = 1e-6; %1e-4;
cfg.con.sig_thr_min = 1e-8;

cfg.con.rcondKKT_min = 1e-12;
cfg.con.kkt_jitter0  = 1e-10;
cfg.con.kkt_jitter_growth = 10;
cfg.con.kkt_jitter_max = 1e-1;
cfg.con.ldl_pivot_min = 1e-14;

%% ------------------------
% DeePC QP weights (THIS is what makes tracking happen)
%% ------------------------
% Qy: output tracking weight per step (scalar or p×p)
% Ru: input weight per step (scalar or m×m)
cfg.deepc.Qy = 1000; 
cfg.deepc.Ru = 1e-3; 
cfg.deepc.alpha = 1e-2;     
cfg.deepc.beta_du = 1e-3;
% penalties used ONLY in the *outer* evaluation objective
cfg.deepc.beta_u = 1e-3;
cfg.deepc.beta_v = 5e1;
cfg.deepc.du_max = inf; % For hard clamp, 0.02~0.1
% Optional output safety penalty (soft constraint)
cfg.deepc.beta_y = 1e3;
cfg.con.enforce_ymax = false;

%% ------------------------
% Verbose/ Printing
%% ------------------------
cfg.print.verbose = true;
cfg.print.eval_print_every = 25;
cfg.print.gen_print_every  = 1;
cfg.print.tail_consistency_check = true; 
cfg.print.best_each_gen    = true;
cfg.print.best_choice      = 'tcheby_to_ideal'; 

%% ------------------------
% Cache
%% ------------------------
cfg.cache.enable = true;

%% ------------------------
% Metrics (PPSN: HV + diversity)
%% ------------------------
cfg.metrics = struct();

% HV reference handling:

cfg.metrics.hv_ref_mode = 'by_model';
cfg.metrics.hv_ref = [];                 

% Normalize objectives by hv_ref before HV computation -> HV in [0,1]
cfg.metrics.hv_normalize = true;
cfg.metrics.store_pop = true;   

%% ------------------------
% Set Tfull consistently with bounds
%% ------------------------
cfg.prob.Tfull = max(cfg.lb(1)+cfg.lb(2)+cfg.lb(3), cfg.ub(1)+cfg.ub(2)+cfg.ub(3)) + 500;

end

