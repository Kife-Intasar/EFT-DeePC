function out = deepc_run(cfg)


% Returns:
%   out.final.X, out.final.F, out.final.CV
%   out.hist(g): feasible_ratio, bestFeas, Fnd, nd_count, cache_count, eval_new, eval_cache
%   out.cfg, out.prob

if nargin<1 || isempty(cfg)
    cfg = deepc_default_cfg();
end

% Backward compatible defaults
if ~isfield(cfg,'seed_data'), cfg.seed_data = cfg.seed; end
if ~isfield(cfg,'seed_algo'), cfg.seed_algo = cfg.seed; end

% IMPORTANT:

rng(cfg.seed_algo);

% Register the local evaluator so all external calls to deepc_eval(x,prob) work
deepc_eval('register', @eval_candidate);

print_run_header(cfg);

% Build / reuse problem (offline dataset fixed per run seed)
prob = make_problem(cfg);

cfg  = prob.cfg;
% --- defensive defaults for cache ---
if ~isfield(cfg,'cache') || isempty(cfg.cache)
    cfg.cache = struct();
end
if ~isfield(cfg.cache,'enable') || isempty(cfg.cache.enable)
    cfg.cache.enable = false;
end
prob.cfg = cfg;

if isfield(cfg,'bounds') && isfield(cfg.bounds,'mode')
    switch lower(cfg.bounds.mode)
        case 'data_driven'
            cfg = data_driven_bounds_from_offline(cfg, prob);
        case 'pilot_data_driven'
            cfg = data_driven_bounds_from_pilot(cfg, prob);
        case 'manual'
            % Keep user-specified bounds exactly as given
            if isfield(cfg.bounds,'lb_manual') && ~isempty(cfg.bounds.lb_manual)
                cfg.lb = cfg.bounds.lb_manual;
            end
            if isfield(cfg.bounds,'ub_manual') && ~isempty(cfg.bounds.ub_manual)
                cfg.ub = cfg.bounds.ub_manual;
            end
        otherwise
            error('Unknown cfg.bounds.mode=%s',cfg.bounds.mode);
    end
    cfg.bounds.lb_used = cfg.lb;
    cfg.bounds.ub_used = cfg.ub;
    prob.cfg = cfg; % keep consistent
else
    if ~isfield(cfg,'bound')|| isempty(cfg.bounds)
        cfg.bounds = struct();
    end
    cfg.bounds.lb_used = cfg.lb;
    cfg.bounds.ub_used = cfg.ub;
    prob.cfg = cfg;
end

cfg.lb(1:3) = round(cfg.lb(1:3));
cfg.ub(1:3) = round(cfg.ub(1:3));

if any(~isfinite(cfg.lb)) || any(~isfinite(cfg.ub)) || any(cfg.lb > cfg.ub)
    error('Invalid search bounds: lb = [%s], ub = [%s]', ...
        num2str(cfg.lb), num2str(cfg.ub));
end



% --- CALIBRATE (with cache OFF) ---
if cfg.con.use_tail_constraint && ( ...
        ~isfield(cfg.con,'e_tail_max') || isempty(cfg.con.e_tail_max) || ~isfinite(cfg.con.e_tail_max))

    cfgCal = prob.cfg;
    cfgCal.cache.enable = false;
    cfgCal.con.use_tail_constraint = false;

    probCal = prob;
    probCal.cfg = cfgCal;
    probCal.cache = [];

    % init stats before calibration calls deepc_eval
    global DEEPC_STATS;
    DEEPC_STATS = struct('new',0,'cache',0,'calls',0);
    global DEEPC_EVAL_LOG;
    DEEPC_EVAL_LOG = struct();
    DEEPC_EVAL_LOG.X  = [];
    DEEPC_EVAL_LOG.F  = [];
    DEEPC_EVAL_LOG.CV = [];

    etmax = calibrate_e_tail_max(probCal, cfgCal, 200, 0.90, 0.10);

    if ~isfinite(etmax) || etmax <= 0
        error('calibrate_e_tail_max returned invalid e_tail_max = %g', etmax);
    end

    cfg.con.e_tail_max = etmax;

    % write back into prob used by the EA
    prob.cfg = cfg;
    prob.e_tail_max = etmax;

    fprintf('Calibrated e_tail_max = %.4g\n', etmax);
end

% --- optional one-shot consistency check ---
if isfield(cfg,'print') && isfield(cfg.print,'tail_consistency_check') && cfg.print.tail_consistency_check
    xdbg = init_pop(1, cfg.lb, cfg.ub);
    xdbg = xdbg(1,:);

    [~,~,dbg1] = eval_candidate(xdbg, prob, 'calibration_mode', true);
    [~,~,dbg2] = eval_candidate(xdbg, prob, 'calibration_mode', false);

    fprintf('tail consistency check: cal=%.6f eval=%.6f diff=%.3e\n', ...
        dbg1.e_tail_agg, dbg2.e_tail_agg, abs(dbg1.e_tail_agg-dbg2.e_tail_agg));
end


% Attach cache (per run)
if cfg.cache.enable
    prob.cache = containers.Map('KeyType','char','ValueType','any');
else
    prob.cache = [];
end

% ---- init stats BEFORE calibration uses deepc_eval ----
global DEEPC_STATS;
DEEPC_STATS = struct('new',0,'cache',0,'calls',0);


fprintf('Udata: max|U|=%g, std(U)=%g\n', max(abs(prob.Udata_full(:))), std(prob.Udata_full(:)));
fprintf('Ydata: max|Y|=%g, std(Y)=%g\n', max(abs(prob.Ydata_full(:))), std(prob.Ydata_full(:)));

% global counters (per run)
global DEEPC_STATS;
DEEPC_STATS = struct('new',0,'cache',0,'calls',0);

tEA = tic;   % offline EA runtime
% Run algorithm
switch lower(cfg.algo)
    case 'nsga2'
        [X,F,CV,hist] = run_nsga2_hist(prob,cfg);
    case 'moead'
        [X,F,CV,hist] = run_moead_hist(prob,cfg);
    case 'random'
        [X,F,CV,hist] = run_random_hist(prob,cfg);
    otherwise
        error('cfg.algo must be nsga2, moead, or random');
end
ea_sec = toc(tEA);

out = struct();
out.final = struct('X',X,'F',F,'CV',CV);
out.hist = hist;
out.cfg = cfg;
out.prob = prob;
out.api = struct();
out.api.rollout = @(xbest, opts) deepc_rollout_api(prob, xbest, opts);
out.api.eval = @deepc_eval;
out.runtime = struct();
out.runtime.ea_sec = ea_sec;
out.runtime.eval_budget = cfg.P*(cfg.G+1);

out.best = struct();

if ~isempty(hist(end).Xnd)
    out.best.ctrl_x = hist(end).bestCtrlX;
    out.best.time_x = hist(end).bestTimeX;
    out.best.pareto_x = hist(end).Xnd;

    % decoded control-best
    xc = hist(end).bestCtrlX;
    out.best.ctrl_decoded = struct( ...
        'Tini', round(xc(1)), ...
        'N', round(xc(2)), ...
        'Delta', round(xc(3)), ...
        'log10_lam_g', xc(4), ...
        'log10_lam_sig', xc(5), ...
        'log10_sig_thr', xc(6), ...
        'lam_g', max(10^(xc(4)), cfg.con.lam_g_min), ...
        'lam_sig', max(10^(xc(5)), cfg.con.lam_sig_min), ...
        'sig_thr', max(10^(xc(6)), cfg.con.sig_thr_min) );

    % decoded time-best
    xt = hist(end).bestTimeX;
    out.best.time_decoded = struct( ...
        'Tini', round(xt(1)), ...
        'N', round(xt(2)), ...
        'Delta', round(xt(3)), ...
        'log10_lam_g', xt(4), ...
        'log10_lam_sig', xt(5), ...
        'log10_sig_thr', xt(6), ...
        'lam_g', max(10^(xt(4)), cfg.con.lam_g_min), ...
        'lam_sig', max(10^(xt(5)), cfg.con.lam_sig_min), ...
        'sig_thr', max(10^(xt(6)), cfg.con.sig_thr_min) );
else
    out.best.ctrl_x = [];
    out.best.time_x = [];
    out.best.pareto_x = [];
    out.best.ctrl_decoded = [];
    out.best.time_decoded = [];
end

global DEEPC_EVAL_LOG
out.eval_log = DEEPC_EVAL_LOG;

end


%% =====================================================================
%%                        EVALUATION (ROBUST)
%% =====================================================================

function [f, cv, det] = eval_candidate(x, prob, varargin)
cfg = prob.cfg;
fixed_base_seed = [];
calibration_mode = false;
if mod(numel(varargin),2) ~= 0
    error('eval_candidate: optional args must be name/value pairs.');
end
for kk = 1:2:numel(varargin)
    switch lower(varargin{kk})
        case 'calibration_mode'
            calibration_mode = logical(varargin{kk+1});
        case 'fixed_base_seed'
            fixed_base_seed = varargin{kk+1};
        otherwise
            error('eval_candidate: unknown option "%s".', varargin{kk});
    end
end

use_tail_constraint_eff = false;
if isfield(cfg,'con') && isfield(cfg.con,'use_tail_constraint') && cfg.con.use_tail_constraint
    use_tail_constraint_eff = true;
end
if calibration_mode
    use_tail_constraint_eff = false;
end

rng_state = rng;
cleanup = onCleanup(@() rng(rng_state));

lb = cfg.lb; ub = cfg.ub;
x(1:3) = round(x(1:3));
x = min(max(x, lb), ub);
x(1:3) = round(x(1:3));
seed_offset = 0;
if isfield(cfg,'scen') && isfield(cfg.scen,'seed_offset') && ~isempty(cfg.scen.seed_offset)
    seed_offset = cfg.scen.seed_offset;
end

fixed_seed_tag = -1;
if ~isempty(fixed_base_seed)
    fixed_seed_tag = round(fixed_base_seed);
end

key = sprintf('%d_%d_%d_%.12g_%.12g_%.12g_%d_%d_%d', ...
    x(1), x(2), x(3), x(4), x(5), x(6), seed_offset, use_tail_constraint_eff,fixed_seed_tag);

use_cache = isfield(cfg,'cache') && isfield(cfg.cache,'enable') && cfg.cache.enable ...
    && isfield(prob,'cache') && ~isempty(prob.cache);

if use_cache && isKey(prob.cache, key)
    outc = prob.cache(key);
    f = outc.f; cv = outc.cv; det = outc.det;
    stats_tick(true);
    return;
end

stats_tick(false);

Tini = x(1); N = x(2); Delta = x(3);
K = Tini + N;
Tdata = K + Delta;

lam_g   = max(10^(x(4)), cfg.con.lam_g_min);
lam_sig = max(10^(x(5)), cfg.con.lam_sig_min);
sig_thr = max(10^(x(6)), cfg.con.sig_thr_min);

Tsrc = size(prob.Udata_full,2);
s0 = 1 + floor((Tsrc - Tdata)/2);
s0 = max(1, min(s0, Tsrc - Tdata + 1));

U = prob.Udata_full(:, s0:s0+Tdata-1);
Y = prob.Ydata_full(:, s0:s0+Tdata-1);

Hu = build_hankel(U, K);
Hy = build_hankel(Y, K);
W0 = [Hu; Hy];

sW = svd(W0,'econ');
rW = sum(sW > prob.tol_rank);
if rW < 1, rW = 1; end
sigma_r0 = sW(rW);
kappaW0  = sW(1)/max(sW(end),eps);

if cfg.obj.use_robust
    S = cfg.obj.S;
else
    S = 1;
end

IAE_s   = zeros(S,1);
U2_s    = zeros(S,1);
Viol_s  = zeros(S,1);
Fail_s  = zeros(S,1);
Xpeak_s = zeros(S,1);
Uviol_s = zeros(S,1);
Yviol_s = zeros(S,1);
Ypeak_s = zeros(S,1);
DU2_s   = zeros(S,1);
e_tail_s = zeros(S,1);
Div_s   = zeros(S,1);
t_all   = [];

if isempty(fixed_base_seed)
    base_seed = 1234 + mod(hash_seed(x), 2^31-1) + seed_offset;
else
    base_seed = round(fixed_base_seed) + seed_offset;
end
sigMin_s = inf(S,1);
kapMax_s = zeros(S,1);

probS = prob;

if ~isfield(probS,'ref_fun') || isempty(probS.ref_fun)
    if isfield(cfg,'prob') && isfield(cfg.prob,'ref_fun') && isa(cfg.prob.ref_fun,'function_handle')
        probS.ref_fun = cfg.prob.ref_fun;
    else
        probS.ref_fun = [];
    end
end

if ~isfield(probS,'ref') || isempty(probS.ref) || size(probS.ref,2) < probS.Ksim
    probS.ref = deepc_extend_ref([], probS.p, probS.Ksim, probS.ref_fun);
else
    probS.ref = deepc_extend_ref(probS.ref, probS.p, probS.Ksim, probS.ref_fun);
end

for s = 1:S
    rng(base_seed + 1000*s);

    x0 = prob.x0_nom + cfg.scen.x0_sigma*randn(prob.n,1);

    if cfg.scen.lambda_phase_jitter
        lambda_shift = randi([0 cfg.scen.lambda_shift_max]);
    else
        lambda_shift = 0;
    end

    met = simulate_closed_loop(probS, Tini, N, W0, lam_g, lam_sig, sig_thr, x0, lambda_shift);

    IAE_s(s)    = met.IAE;
    U2_s(s)     = met.u2sum;
    Viol_s(s)   = met.viol;
    Uviol_s(s)  = met.viol;
    Yviol_s(s)  = met.y_viol;
    Ypeak_s(s)  = met.y_peak;
    Fail_s(s)   = met.fail_rate;
    Xpeak_s(s)  = met.x_peak;
    sigMin_s(s) = met.sigma_r_min;
    kapMax_s(s) = met.kappa_max;
    DU2_s(s)    = met.du2sum;
    Div_s(s)    = met.diverged;

    t_ctrl = met.t_solve + met.t_rebuild;
    t_all = [t_all; t_ctrl(:)];

    e_tail_s(s) = met.e_tail;
end

if isfield(cfg.obj,'ctrl_stat') && strcmpi(cfg.obj.ctrl_stat,'max')
    statIAE = max(IAE_s);
else
    statIAE = mean(IAE_s);
end

meanU2    = mean(U2_s);
meanUviol = mean(Uviol_s);
meanYviol = mean(Yviol_s);

if cfg.obj.use_cvar
    cvarIAE = cvar_tail(IAE_s, cfg.obj.q);
else
    cvarIAE = 0;
end

beta_y = 0;
if isfield(cfg,'deepc') && isfield(cfg.deepc,'beta_y') && ~isempty(cfg.deepc.beta_y)
    beta_y = cfg.deepc.beta_y;
end

Jctrl = statIAE + cfg.obj.alpha_cvar*cvarIAE ...
    + prob.beta_u*meanU2 ...
    + prob.beta_v*meanUviol ...
    + beta_y*meanYviol;

switch lower(cfg.obj.time_stat)
    case 'median'
        t_stat = median(t_all);
    case 'mean'
        t_stat = mean(t_all);
    case 'max'
        t_stat = max(t_all);
    otherwise
        error('Unknown cfg.obj.time_stat = %s', cfg.obj.time_stat);
end

if cfg.obj.use_cvar
    cvarT = cvar_tail(t_all, cfg.obj.q);
else
    cvarT = 0;
end

Jtime = t_stat + cfg.obj.gamma_time*cvarT;
Jtime = 1e3 * Jtime;

f = [Jctrl, Jtime];

fail_max        = max(Fail_s);
xpeak_max       = max(Xpeak_s);
y_peak_max      = max(Ypeak_s);
div_any         = max(Div_s);
kappa_max_all   = max(kapMax_s);
sigma_r_min_all = min(sigMin_s);

cv1 = max(0, log10(kappa_max_all / prob.kappa_max));
cv2 = max(0, (prob.sigma_min - sigma_r_min_all) / prob.sigma_min);
cv3 = max(0, (fail_max - prob.fail_rate_max) / prob.fail_rate_max);
cv4 = max(0, (xpeak_max - prob.x_max) / prob.x_max);
cv5 = div_any;

cv_y = 0;
if isfield(cfg,'con') && isfield(cfg.con,'enforce_ymax') && cfg.con.enforce_ymax ...
        && isfield(prob,'y_max') && isfinite(prob.y_max)
    cv_y = max(0, (y_peak_max - prob.y_max) / max(prob.y_max, eps));
end

cv = cv1 + cv2 + cv3 + cv4 + cv5 + cv_y;

e_tail_agg = NaN;
cv_track   = 0;
e_tail_max = NaN;

tail_stat = 'mean';
if isfield(cfg,'con') && isfield(cfg.con,'e_tail_stat') && ~isempty(cfg.con.e_tail_stat)
    tail_stat = cfg.con.e_tail_stat;
end

et = e_tail_s(isfinite(e_tail_s));

if ~isempty(et)
    switch lower(tail_stat)
        case 'max'
            e_tail_agg = max(et);
        case 'mean'
            e_tail_agg = mean(et);
        case 'cvar'
            e_tail_agg = cvar_tail(et, cfg.obj.q);
        otherwise
            error('Unknown cfg.con.e_tail_stat = %s', tail_stat);
    end
end

if use_tail_constraint_eff
    if ~isfield(cfg,'con') || ~isfield(cfg.con,'e_tail_max') || isempty(cfg.con.e_tail_max) || ~isfinite(cfg.con.e_tail_max)
        error(['Tail constraint enabled in eval_candidate, but cfg.con.e_tail_max is not finite. ' ...
            'Run calibration first or call eval_candidate(..., ''calibration_mode'', true).']);
    end

    e_tail_max = cfg.con.e_tail_max;

    if ~isfinite(e_tail_agg)
        cv_track = 1;
    else
        cv_track = max(0, (e_tail_agg - e_tail_max) / max(e_tail_max, eps));
    end

    cv = cv + cv_track;
end

det = struct();
det.x = x;
det.f = f;
det.cv = cv;
det.is_feasible = (cv <= 0);
det.calibration_mode = calibration_mode;
det.tail_constraint_enforced = use_tail_constraint_eff;

det.sigma_r0 = sigma_r0;
det.kappaW0  = kappaW0;
det.IAE_s    = IAE_s;
det.U2_s     = U2_s;
det.Viol_s   = Viol_s;
det.Fail_s   = Fail_s;
det.Xpeak_s  = Xpeak_s;

det.e_tail_s   = e_tail_s;
det.e_tail_agg = e_tail_agg;
det.cv_track   = cv_track;

det.meanIAE  = statIAE;
det.cvarIAE  = cvarIAE;
det.time_stat = t_stat;
det.cvarT     = cvarT;

det.Tini  = Tini;
det.N     = N;
det.Delta = Delta;
det.K     = K;
det.L     = Delta + 1;
det.Tdata = Tdata;

det.rowsW = (prob.m + prob.p)*K;
det.colsW = det.L;

det.fixed_base_seed = fixed_base_seed;
det.ra_first      = met.ra_first;
det.kappa_max     = kappa_max_all;
det.sigma_r_min   = sigma_r_min_all;
det.y_peak_max    = y_peak_max;
det.diverged_any  = div_any;

det.cv_parts = struct( ...
    'cv1_cond',  cv1, ...
    'cv2_sigma', cv2, ...
    'cv3_fail',  cv3, ...
    'cv4_xpeak', cv4, ...
    'cv5_div',   cv5, ...
    'cv_y',      cv_y, ...
    'cv_track',  cv_track, ...
    'cv_total',  cv);

det.fail_max    = fail_max;
det.xpeak_max   = xpeak_max;
det.y_peak_max  = y_peak_max;

det.kappa_limit = prob.kappa_max;
det.sigma_limit = prob.sigma_min;
det.fail_limit  = prob.fail_rate_max;
det.x_limit     = prob.x_max;

if isfield(prob,'y_max') && isfinite(prob.y_max)
    det.y_limit = prob.y_max;
else
    det.y_limit = NaN;
end

if isfield(cfg,'con') && isfield(cfg.con,'e_tail_max') && isfinite(cfg.con.e_tail_max)
    det.e_tail_max = cfg.con.e_tail_max;
elseif isfield(prob,'e_tail_max') && isfinite(prob.e_tail_max)
    det.e_tail_max = prob.e_tail_max;
else
    det.e_tail_max = NaN;
end


global DEEPC_EVAL_LOG;
DEEPC_EVAL_LOG = struct();
DEEPC_EVAL_LOG.X  = [];
DEEPC_EVAL_LOG.F  = [];
DEEPC_EVAL_LOG.CV = [];

if use_cache
    prob.cache(key) = struct('f',f,'cv',cv,'det',det);
end

cacheN = 0;
if use_cache
    cacheN = prob.cache.Count;
end

if isfield(cfg,'print') && isfield(cfg.print,'verbose') && cfg.print.verbose
    global DEEPC_STATS;
    if mod(DEEPC_STATS.new, cfg.print.eval_print_every)==0 && DEEPC_STATS.new>0
        fprintf('[eval new=%d cache=%d] x=[%d %d %d %.2f %.2f %.2f] J=[%.3g %.3g] CV=%.3g cacheN=%d\n', ...
            DEEPC_STATS.new, DEEPC_STATS.cache, x(1),x(2),x(3),x(4),x(5),x(6), f(1),f(2),cv, cacheN);
    end
end

if use_tail_constraint_eff
    fprintf('TAIL: e_tail_agg=%.4g  e_tail_max=%.4g  cv_track=%.4g\n', ...
        e_tail_agg, e_tail_max, cv_track);
end
end

function met = simulate_closed_loop(prob, Tini, N, W0, lam_g, lam_sig, sig_thr, x0, lambda_shift, doLog)

if nargin < 10 || isempty(doLog)
    doLog = false;
end

cfg = prob.cfg;

n = prob.n;
m = prob.m;
p = prob.p;

if ~isfield(prob,'ref_fun') || isempty(prob.ref_fun)
    if isfield(cfg,'prob') && isfield(cfg.prob,'ref_fun') && isa(cfg.prob.ref_fun,'function_handle')
        prob.ref_fun = cfg.prob.ref_fun;
    else
        prob.ref_fun = [];
    end
end

if ~isfield(prob,'ref') || isempty(prob.ref) || size(prob.ref,2) < prob.Ksim
    prob.ref = deepc_extend_ref([], p, prob.Ksim, prob.ref_fun);
else
    prob.ref = deepc_extend_ref(prob.ref, p, prob.Ksim, prob.ref_fun);
end

xk = x0;
u_hist = zeros(m,Tini);
y_hist = zeros(p,Tini);
u_prev = zeros(m,1);

% warmup
for k = 1:Tini
    y_hist(:,k) = prob.C*xk + deepc_bounded_noise(prob.dm_max,p);
    [A,B] = deepc_AB_of_k(prob,k,lambda_shift);
    xk = A*xk + B*u_hist(:,k) + deepc_bounded_noise(prob.dp_max,n);
end

err_abs = nan(prob.Ksim,1);
u2      = zeros(prob.Ksim,1);
du2     = zeros(prob.Ksim,1);

t_solve   = zeros(prob.Ksim,1);
t_rebuild = zeros(prob.Ksim,1);

viol = 0;
y_viol = 0;
y_peak = 0;
y_max = inf;
if isfield(prob,'y_max') && ~isempty(prob.y_max)
    y_max = prob.y_max;
end

fails = 0;
x_peak = norm(xk);
diverged = false;

W = W0;

sigma_r_min = inf;
kappa_max   = 0;
ra_first = NaN;

need_rebuild = true;
kkt = [];

if doLog
    x_log   = nan(n, prob.Ksim+1);
    y_log   = nan(p, prob.Ksim);
    u_log   = nan(m, prob.Ksim);
    lam_log = nan(1, prob.Ksim);
    ra_log  = nan(1, prob.Ksim);
    L_log   = nan(1, prob.Ksim);
    r_log   = nan(p, prob.Ksim);

    x_log(:,1) = xk;
    ra_hold = NaN;
end

t_end = 0;

for t = 1:prob.Ksim

    if need_rebuild
        tr = tic;

        sW = svd(W,'econ');
        rW = sum(sW > prob.tol_rank);
        if rW < 1, rW = 1; end
        sigma_r = sW(rW);
        kappaW  = sW(1)/max(sW(end),eps);

        sigma_r_min = min(sigma_r_min, sigma_r);
        kappa_max   = max(kappa_max,   kappaW);

        [basis, ra] = build_deepc_basis(W, sig_thr, prob, cfg, Tini, N);
        if isnan(ra_first), ra_first = ra; end

        kkt = factorize_deepc_kkt(basis.Up,basis.Uf,basis.Yp,basis.Yf, ...
            lam_g, lam_sig, m,p,Tini,N, cfg);

        t_rebuild(t) = toc(tr);
        need_rebuild = false;

        if doLog
            ra_hold = ra;
        end
    end

    if doLog
        ra_log(t) = ra_hold;
        lam_log(t) = deepc_lambda_profile(t+Tini+lambda_shift);
        L_log(t) = size(W,2);
    end

    u_ini = reshape(u_hist,[],1);
    y_ini = reshape(y_hist,[],1);
    beq   = [u_ini; y_ini];

    ts = tic;
    r_step = deepc_ref_step_prob(prob, t);

    if isfield(cfg,'print') && isfield(cfg.print,'debug_ref') && cfg.print.debug_ref && t==1
        fprintf('DEBUG r_step(1) = [%g %g]\n', r_step(1), r_step(min(2,end)));
    end

    rN = repmat(r_step, N, 1);
    kkt.rhs_top(1:kkt.ra) = 2 * (kkt.YfTQ * rN);

    [u0, ok] = solve_deepc_kkt_cached(kkt, beq);
    t_solve(t) = toc(ts);

    if t==1
        fprintf('DEBUG u0=[%.4f %.4f], ok=%d, rcond=%.2e, ra=%d\n', ...
            u0(1), u0(min(2,end)), ok, kkt.rcondKKT, kkt.ra);
    end

    if ~ok || any(~isfinite(u0))
        fails = fails + 1;
        u = zeros(m,1);
    else
        u = min(max(u0, prob.umin), prob.umax);
        viol = viol + sum(abs(u-u0));
    end

    if isfield(cfg,'deepc') && isfield(cfg.deepc,'du_max') && ...
            isfinite(cfg.deepc.du_max) && (cfg.deepc.du_max > 0)
        du = u - u_prev;
        du = min(max(du, -cfg.deepc.du_max), cfg.deepc.du_max);
        u  = u_prev + du;
        u  = min(max(u, prob.umin), prob.umax);
    end

    y = prob.C*xk + deepc_bounded_noise(prob.dm_max,p);
    ay = abs(y);
    y_peak = max(y_peak, max(ay));

    if isfinite(y_max)
        y_viol = y_viol + sum(max(0, ay - y_max));
    end

    [A,B] = deepc_AB_of_k(prob, t+Tini, lambda_shift);
    xk = A*xk + B*u + deepc_bounded_noise(prob.dp_max,n);

    x_peak = max(x_peak, norm(xk));

    ref_scale = abs(r_step(:));
    ref_floor = 0.1 * max(abs(prob.ref(:,1:prob.Ksim)), [], 2);

    if isempty(ref_floor) || numel(ref_floor) ~= p || any(~isfinite(ref_floor))
        ref_floor = ones(p,1);
    end

    ys = max(ref_scale, ref_floor);
    ys = max(ys, 1e-8);

    err_abs(t) = mean(abs(y - r_step)./ys);

    u2(t) = norm(u,2)^2;
    du = u - u_prev;
    du2(t) = norm(du,2)^2;
    u_prev = u;
    t_end = t;

    if doLog
        u_log(:,t) = u;
        y_log(:,t) = y;
        r_log(:,t) = r_step;
        x_log(:,t+1) = xk;
    end

    if mod(t, cfg.online.update_period)==0 && ~strcmpi(cfg.online.update_mode,'never')
        wcol  = make_recent_column(u_hist, y_hist, m, p, Tini, N);
        W_try = [W, wcol];

        Wcap = inf;
        if isfield(cfg,'online') && isfield(cfg.online,'window_max_cols') && ~isempty(cfg.online.window_max_cols)
            Wcap = cfg.online.window_max_cols;
        end
        if isfinite(Wcap) && size(W_try,2) > Wcap
            W_try = W_try(:, end-Wcap+1:end);
        end

        if strcmpi(cfg.online.update_mode,'always')
            W = W_try;
            need_rebuild = true;
        else
            sTry = svd(W_try,'econ');
            rTry = sum(sTry > prob.tol_rank);
            if rTry < 1, rTry = 1; end
            sigma_r_try = sTry(rTry);
            if sigma_r_try >= sig_thr
                W = W_try;
                need_rebuild = true;
            end
        end
    end

    u_hist = [u_hist(:,2:end), u];
    y_hist = [y_hist(:,2:end), y];

    if ~isfinite(x_peak) || x_peak > 1e6
        diverged = true;
        fails = fails + (prob.Ksim - t);
        break;
    end
end

errv = err_abs(1:t_end);
errv = errv(isfinite(errv));

if isempty(errv)
    met.IAE = inf;
else
    met.IAE = sum(errv);
end

met.u2sum     = sum(u2);
met.viol      = viol;
met.fail_rate = fails/prob.Ksim;
met.x_peak    = x_peak;
met.y_viol    = y_viol;
met.y_peak    = y_peak;

met.t_solve   = t_solve;
met.t_rebuild = t_rebuild;
met.diverged  = diverged;

met.sigma_r_min = sigma_r_min;
met.kappa_max   = kappa_max;
met.ra_first    = ra_first;
met.du2sum      = sum(du2);

tailFrac = 0.20;
tailMin  = 10;
if isfield(cfg,'con') && isfield(cfg.con,'e_tail_frac'), tailFrac = cfg.con.e_tail_frac; end
if isfield(cfg,'con') && isfield(cfg.con,'e_tail_min_steps'), tailMin = cfg.con.e_tail_min_steps; end

Tuse = numel(errv);
if Tuse == 0
    met.e_tail = inf;
else
    tail = max(tailMin, round(tailFrac * Tuse));
    tail = min(tail, Tuse);
    met.e_tail = mean(errv(end-tail+1:end));
end

if doLog
    met.x_log   = x_log;
    met.y_log   = y_log;
    met.u_log   = u_log;
    met.lam_log = lam_log;
    met.ra_log  = ra_log;
    met.L_log   = L_log;
    met.r_log   = r_log;
end
end


function wcol = make_recent_column(u_hist, y_hist, m, p, Tini, N)
uK = [reshape(u_hist,[],1); zeros(m*N,1)];
yK = [reshape(y_hist,[],1); zeros(p*N,1)];
wcol = [uK; yK];
end

function H = build_hankel(sig, K)
[d,T] = size(sig);
L = T - K + 1;
H = zeros(d*K, L);
for i=1:K
    H((i-1)*d+1:i*d,:) = sig(:, i:i+L-1);
end
end

function c = cvar_tail(x, q)
x = x(:); x = x(isfinite(x));
n = numel(x);
if n==0, c=1e9; return; end
x = sort(x,'ascend');
k = max(1, min(n, ceil(q*n)));
c = mean(x(k:end));
end

function h = hash_seed(x)
x = x(:);
h = 0;
for i=1:numel(x)
    h = h + floor(1e6*abs(sin(1000*x(i)+17*i))) * i;
end
h = mod(h, 2^31-1);
end

function stats_tick(isCacheHit)
global DEEPC_STATS;

if isempty(DEEPC_STATS) || ~isstruct(DEEPC_STATS) ...
        || ~all(isfield(DEEPC_STATS, {'new','cache','calls'}))
    DEEPC_STATS = struct('new',0,'cache',0,'calls',0);
end

DEEPC_STATS.calls = DEEPC_STATS.calls + 1;
if isCacheHit
    DEEPC_STATS.cache = DEEPC_STATS.cache + 1;
else
    DEEPC_STATS.new = DEEPC_STATS.new + 1;
end
end

%% =====================================================================
%%                      HISTORY METRICS (PER GEN)
%% =====================================================================

function h = record_history(gen, prob, X, F, CV)
cfg = prob.cfg;

feas = (CV<=0);
h = struct();
h.gen = gen;
h.feasible_ratio = mean(feas);

% feasible ND front
[Fnd, Xnd] = feasible_nondominated(F,CV,X);
h.Fnd = Fnd;
h.Xnd = Xnd;
h.nd_count = size(Fnd,1);

hv_ref = [1e3 1e2];
if isfield(cfg,'metrics') && isfield(cfg.metrics,'hv_ref') && ~isempty(cfg.metrics.hv_ref)
    hv_ref = cfg.metrics.hv_ref;
end

if isempty(h.Fnd)
    h.hv = 0;
    h.spacing = NaN;
    h.spread  = 0;
else
    Fn = h.Fnd;

    if isfield(cfg,'metrics') && isfield(cfg.metrics,'hv_normalize') && cfg.metrics.hv_normalize
        FnN  = Fn ./ max(hv_ref, eps);
        refN = [1 1];
    else
        FnN  = Fn;
        refN = hv_ref;
    end

    h.hv      = hv2d_min(FnN, refN);
    h.spacing = spacing_metric(FnN);
    h.spread  = max_spread(FnN, refN);
end

if isfield(cfg,'metrics') && isfield(cfg.metrics,'store_pop') && cfg.metrics.store_pop
    h.Xpop = X;
    h.Fpop = F;
    h.CVpop = CV;
end

if any(feas)
    feas_idx = find(feas);

    [~, kT] = min(F(feas,2));
    ibT = feas_idx(kT);
    h.bestTimeFeas = F(ibT,:);
    h.bestTimeX    = X(ibT,:);

    [~, kC] = min(F(feas,1));
    ibC = feas_idx(kC);
    h.bestCtrlFeas = F(ibC,:);
    h.bestCtrlX    = X(ibC,:);

    oldCache = prob.cfg.cache.enable;
    prob.cfg.cache.enable = false;

    [~, ~, detT] = deepc_eval(h.bestTimeX, prob);
    [~, ~, detC] = deepc_eval(h.bestCtrlX, prob);

    prob.cfg.cache.enable = oldCache;

    if isfield(detT,'e_tail_agg')
        h.bestTimeTail = detT.e_tail_agg;
    else
        h.bestTimeTail = NaN;
    end

    if isfield(detC,'e_tail_agg')
        h.bestCtrlTail = detC.e_tail_agg;
    else
        h.bestCtrlTail = NaN;
    end

    if isfield(detT,'cv_track')
        h.bestTimeCVtrack = detT.cv_track;
    else
        h.bestTimeCVtrack = NaN;
    end

    if isfield(detC,'cv_track')
        h.bestCtrlCVtrack = detC.cv_track;
    else
        h.bestCtrlCVtrack = NaN;
    end

    h.bestFeas = h.bestTimeFeas;

else
    h.bestTimeFeas   = [nan nan];
    h.bestCtrlFeas   = [nan nan];
    h.bestFeas       = [nan nan];

    h.bestTimeX      = nan(1,size(X,2));
    h.bestCtrlX      = nan(1,size(X,2));

    h.bestTimeTail   = NaN;
    h.bestCtrlTail   = NaN;
    h.bestTimeCVtrack = NaN;
    h.bestCtrlCVtrack = NaN;
end

if cfg.cache.enable
    h.cache_count = prob.cache.Count;
else
    h.cache_count = 0;
end

% eval counters
global DEEPC_STATS;
h.eval_new = DEEPC_STATS.new;
h.eval_cache = DEEPC_STATS.cache;

if cfg.print.best_each_gen && mod(gen, cfg.print.gen_print_every)==0
    print_best_candidate(gen, prob, X, F, CV);
end

end

function [Fnd, Xnd] = feasible_nondominated(F,CV,X)
feas = (CV<=0);
if ~any(feas)
    Fnd = zeros(0,2);
    Xnd = zeros(0,size(X,2));
    return;
end
Ff = F(feas,:);
Xf = X(feas,:);
nd = nondominated_mask(Ff);
Fnd = Ff(nd,:);
Xnd = Xf(nd,:);

% Remove duplicate decision vectors
[Xnd_unique, ia, ~] = unique(Xnd, 'rows', 'stable');
Fnd = Fnd(ia,:);
Xnd = Xnd_unique;
end

function nd = nondominated_mask(F)
N = size(F,1);
nd = true(N,1);
for i=1:N
    if ~nd(i), continue; end
    for j=1:N
        if i==j, continue; end
        if all(F(j,:)<=F(i,:)) && any(F(j,:)<F(i,:))
            nd(i) = false;
            break;
        end
    end
end
end

function print_best_candidate(gen, prob, X, F, CV)
cfg = prob.cfg;

feas = (CV<=0);

if ~any(feas)
    [cvmin, ib_bad] = min(CV);
    fprintf('  [best@gen %3d] no feasible candidate | least-infeasible x=%s CV=%.4g J=[%.4g %.4g]\n', ...
        gen, xfmt(X(ib_bad,:)), cvmin, F(ib_bad,1), F(ib_bad,2));
    return;
end

idx = find(feas);
Ff  = F(feas,:);

% FAST: min time
[~,kT] = min(Ff(:,2)); ib_fast = idx(kT);

% CTRL: min control
[~,kC] = min(Ff(:,1)); ib_ctrl = idx(kC);

% COMP: Tchebycheff to ideal Z
Z = min(Ff,[],1);
vals = zeros(size(Ff,1),1);
for i=1:size(Ff,1)
    vals(i) = max(abs(Ff(i,:) - Z));
end
[~,kX] = min(vals); ib_comp = idx(kX);

fprintf('  [gen %3d] FAST x=%s J=[%.4g %.4g]\n', gen, xfmt(X(ib_fast,:)), F(ib_fast,1), F(ib_fast,2));
fprintf('           CTRL x=%s J=[%.4g %.4g]\n',     xfmt(X(ib_ctrl,:)), F(ib_ctrl,1), F(ib_ctrl,2));
fprintf('           COMP x=%s J=[%.4g %.4g]\n',     xfmt(X(ib_comp,:)), F(ib_comp,1), F(ib_comp,2));

% choose which gets per-scenario detail
detail_mode = 'fast';

switch lower(detail_mode)
    case 'fast'
        ib = ib_fast;
    case 'ctrl'
        ib = ib_ctrl;
    case 'comp'
        ib = ib_comp;
    otherwise
        switch lower(cfg.print.best_choice)
            case 'fastest_feasible'
                ib = ib_fast;
            case 'min_sum'
                [~,k] = min(sum(Ff,2)); ib = idx(k);
            otherwise
                ib = ib_fast;
        end
end

xbest = X(ib,:);

fprintf('            (selected index=%d) rawF=[%.4g %.4g]\n', ib, F(ib,1), F(ib,2));

[fb, cvb, det] = eval_candidate(xbest, prob);

cp = det.cv_parts;
fprintf('               CV parts: cond=%.3g sigma=%.3g fail=%.3g xpeak=%.3g div=%.3g y=%.3g tail=%.3g | TOTAL=%.3g\n', ...
    cp.cv1_cond, cp.cv2_sigma, cp.cv3_fail, cp.cv4_xpeak, cp.cv5_div, cp.cv_y, cp.cv_track, cp.cv_total);

fprintf('  [detail@gen %3d] x=[%d %d %d %.3f %.3f %.3f] J=[%.4g %.4g] CV=%.3g\n', ...
    gen, det.x(1),det.x(2),det.x(3),det.x(4),det.x(5),det.x(6), fb(1),fb(2),cvb);

fprintf('               cond: sigma_r0=%.3g kappaW0=%.3g | robust: meanIAE=%.3g cvarIAE=%.3g timeStat=%.3g cvarT=%.3g\n', ...
    det.sigma_r0, det.kappaW0, det.meanIAE, det.cvarIAE, det.time_stat, det.cvarT);

fprintf('               per-scenario:   IAE     fail    xpeak    viol     U2\n');
for s=1:numel(det.IAE_s)
    fprintf('                 s=%2d      %7.3g  %6.3f  %7.3g  %7.3g  %7.3g\n', ...
        s, det.IAE_s(s), det.Fail_s(s), det.Xpeak_s(s), det.Viol_s(s), det.U2_s(s));
end

end


function s = xfmt(x)
x(1:3) = round(x(1:3));
s = sprintf('[%d %d %d %.3f %.3f %.3f]', x(1),x(2),x(3),x(4),x(5),x(6));
end

%% =====================================================================
%%                         NSGA-II (HISTORY)
%% =====================================================================

function [X,F,CV,hist] = run_nsga2_hist(prob,cfg)
P = cfg.P; G = cfg.G;

X = init_pop(P,cfg.lb,cfg.ub);
[F,CV] = eval_pop(prob,X);

h0 = record_history(0,prob,X,F,CV);
hist = repmat(h0, G+1, 1);
hist(1) = h0;

for gen=1:G
    Xc = zeros(P, size(X,2));
    for i=1:2:P
        a = tournament_select(F,CV);
        b = tournament_select(F,CV);
        p1 = X(a,:); p2 = X(b,:);

        if rand < cfg.nsga2.pc
            [c1,c2] = sbx_mixed(p1,p2,cfg.lb,cfg.ub,cfg.nsga2.eta_c);
        else
            c1=p1; c2=p2;
        end

        c1 = poly_mut_mixed(c1,cfg.lb,cfg.ub,cfg.nsga2.pm,cfg.nsga2.eta_m);
        c2 = poly_mut_mixed(c2,cfg.lb,cfg.ub,cfg.nsga2.pm,cfg.nsga2.eta_m);

        Xc(i,:) = c1;
        if i+1<=P, Xc(i+1,:)=c2; end
    end

    [Fc,CVc] = eval_pop(prob,Xc);

    Xall=[X;Xc]; Fall=[F;Fc]; CVall=[CV;CVc];
    [rank,crowd] = nsga2_rank_crowd(Fall,CVall);
    [~,idx] = sortrows([rank,-crowd],[1 2]);
    idx = idx(1:P);

    X=Xall(idx,:); F=Fall(idx,:); CV=CVall(idx);

    cacheN = 0;
    if cfg.cache.enable
        cacheN = prob.cache.Count;
    end

    global DEEPC_STATS;
    hitRate = 100 * (DEEPC_STATS.cache / max(1,DEEPC_STATS.calls));

    feasN = sum(CV<=0);

    if feasN>0
        [bestT,kk] = min(F(CV<=0,2));
        feas_idx = find(CV<=0);
        ib = feas_idx(kk);
        fprintf('[%s gen %3d/%3d] feas=%3d/%3d | bestFeas J=[%.3g %.3g] | evalNew=%d cacheHit=%d (%.1f%%) | cacheN=%d\n', ...
            upper(cfg.algo), gen, cfg.G, feasN, cfg.P, F(ib,1), bestT, ...
            DEEPC_STATS.new, DEEPC_STATS.cache, hitRate, cacheN);
    else
        fprintf('[%s gen %3d/%3d] feas=%3d/%3d | evalNew=%d cacheHit=%d (%.1f%%) | cacheN=%d\n', ...
            upper(cfg.algo), gen, cfg.G, feasN, cfg.P, ...
            DEEPC_STATS.new, DEEPC_STATS.cache, hitRate, cacheN);
    end
    hist(gen+1) = record_history(gen,prob,X,F,CV);
end
end

function [rank,crowd] = nsga2_rank_crowd(F,CV)
N=size(F,1);
rank=inf(N,1); crowd=zeros(N,1);
feas = (CV<=0);

if any(feas)
    Ff=F(feas,:);
    fronts = fast_nondom_sort(Ff);
    idxf = find(feas);
    for k=1:numel(fronts)
        rank(idxf(fronts{k}))=k;
        crowd(idxf(fronts{k}))=crowding_distance(Ff(fronts{k},:));
    end
end

if any(~feas)
    idxi=find(~feas);
    [~,ord]=sort(CV(~feas),'ascend');
    maxr=max(rank(feas)); if isempty(maxr), maxr=0; end
    for j=1:numel(ord)
        rank(idxi(ord(j))) = maxr + 1 + j;
        crowd(idxi(ord(j)))=0;
    end
end
end

function fronts = fast_nondom_sort(F)
N=size(F,1);
S=cell(N,1); n=zeros(N,1);
fronts={}; F1=[];
for p=1:N
    Sp=[]; np=0;
    for q=1:N
        if p==q, continue; end
        if dominates(F(p,:),F(q,:)), Sp(end+1)=q;
        elseif dominates(F(q,:),F(p,:)), np=np+1;
        end
    end
    S{p}=Sp; n(p)=np;
    if np==0, F1(end+1)=p;
    end
end
fronts{1}=F1;
i=1;
while ~isempty(fronts{i})
    Q=[];
    for p=fronts{i}
        for q=S{p}
            n(q)=n(q)-1;
            if n(q)==0, Q(end+1)=q;
            end
        end
    end
    i=i+1; fronts{i}=Q;
end
if isempty(fronts{end}), fronts(end)=[]; end
end

function cd = crowding_distance(F)
n=size(F,1); m=size(F,2);
cd=zeros(n,1);
if n<=2, cd(:)=inf; return; end
for j=1:m
    [~,idx]=sort(F(:,j));
    cd(idx(1))=inf; cd(idx(end))=inf;
    fmin=F(idx(1),j); fmax=F(idx(end),j);
    if fmax-fmin<eps, continue; end
    for i=2:n-1
        cd(idx(i)) = cd(idx(i)) + (F(idx(i+1),j)-F(idx(i-1),j))/(fmax-fmin);
    end
end
end

function tf = dominates(a,b)
tf = all(a<=b) && any(a<b);
end

function k = tournament_select(F,CV)
N=size(F,1);
a=randi(N); b=randi(N);
fa=(CV(a)<=0); fb=(CV(b)<=0);
if fa && ~fb, k=a; return; end
if fb && ~fa, k=b; return; end
if ~fa && ~fb
    if CV(a)<CV(b), k=a; else, k=b; end
    return;
end
if dominates(F(a,:),F(b,:)), k=a;
elseif dominates(F(b,:),F(a,:)), k=b;
else
    if sum(F(a,:))<sum(F(b,:)), k=a; else, k=b; end
end
end

%% =====================================================================
%%                         MOEA/D (HISTORY)
%% =====================================================================

function [X,F,CV,hist] = run_moead_hist(prob,cfg)
P = cfg.P; G = cfg.G;

% weights
W = zeros(P,2);
for i=1:P
    w1=(i-1)/(P-1);
    W(i,:)=[w1,1-w1];
end

% neighborhoods
Tn = cfg.moead.Tn;
D=zeros(P,P); B=zeros(P,Tn);
for i=1:P
    for j=1:P, D(i,j)=norm(W(i,:)-W(j,:)); end
    [~,idx]=sort(D(i,:),'ascend');
    B(i,:)=idx(1:Tn);
end

X = init_pop(P,cfg.lb,cfg.ub);
[F,CV] = eval_pop(prob,X);
Z = min(F,[],1);

scale = cfg.metrics.hv_ref(:).';

h0 = record_history(0,prob,X,F,CV);
hist = repmat(h0, G+1, 1);
hist(1) = h0;

for gen=1:G
    for i=1:P
        Pn=B(i,:);
        r=Pn(randperm(numel(Pn),3));

        y = de_mixed(X(r(1),:),X(r(2),:),X(r(3),:),cfg.lb,cfg.ub,cfg.moead.F,cfg.moead.CR);
        y = poly_mut_mixed(y,cfg.lb,cfg.ub,cfg.moead.pm,cfg.moead.eta_m);

        [fy,cvy] = eval_candidate(y,prob);

        Z = min(Z,fy);

        for j=1:Tn
            k = B(i,j);

            if (cvy<=0) && (CV(k)>0)
                X(k,:)=y; F(k,:)=fy; CV(k)=cvy;
            elseif (cvy>0) && (CV(k)>0)
                if cvy < CV(k)
                    X(k,:)=y; F(k,:)=fy; CV(k)=cvy;
                end
            else

                gold = tcheby(F(k,:),W(k,:),Z,scale);
                gnew = tcheby(fy,    W(k,:),Z,scale);
                if gnew <= gold
                    X(k,:)=y; F(k,:)=fy; CV(k)=cvy;
                end
            end
        end
    end

    cacheN = 0;
    if cfg.cache.enable
        cacheN = prob.cache.Count;
    end

    global DEEPC_STATS;
    hitRate = 100 * (DEEPC_STATS.cache / max(1,DEEPC_STATS.calls));

    feasN = sum(CV<=0);
    if feasN>0
        [bestT,kk] = min(F(CV<=0,2));
        feas_idx = find(CV<=0);
        ib = feas_idx(kk);
        fprintf('[%s gen %3d/%3d] feas=%3d/%3d | bestFeas J=[%.3g %.3g] | evalNew=%d cacheHit=%d (%.1f%%) | cacheN=%d\n', ...
            upper(cfg.algo), gen, cfg.G, feasN, cfg.P, F(ib,1), bestT, ...
            DEEPC_STATS.new, DEEPC_STATS.cache, hitRate, cacheN);
    else
        fprintf('[%s gen %3d/%3d] feas=%3d/%3d | evalNew=%d cacheHit=%d (%.1f%%) | cacheN=%d\n', ...
            upper(cfg.algo), gen, cfg.G, feasN, cfg.P, ...
            DEEPC_STATS.new, DEEPC_STATS.cache, hitRate, cacheN);
    end

    hist(gen+1) = record_history(gen,prob,X,F,CV);
end
end



function g=tcheby(f,w,z,s)
w=max(w,1e-6);
s=max(s,1e-9);
g=max(w.*abs((f-z)./s));
end

function y=de_mixed(x1,x2,x3,lb,ub,F,CR)
v=x1 + F*(x2-x3);
y=x1;
jrand=randi(numel(x1));
for j=1:numel(x1)
    if rand<CR || j==jrand, y(j)=v(j); end
end
y=min(max(y,lb),ub);
y(1:3)=round(y(1:3));
y(1:3)=min(max(y(1:3),lb(1:3)),ub(1:3));
end

%% =====================================================================
%%                         Random (HISTORY)
%% =====================================================================

function [X,F,CV,hist] = run_random_hist(prob,cfg)


P = cfg.P; G = cfg.G;


% preallocate hist
dummyX = init_pop(P, cfg.lb, cfg.ub);
[dF,dCV] = eval_pop(prob,dummyX);
h0 = record_history(0, prob, dummyX, dF, dCV);
hist = repmat(h0, G+1, 1);

Xall = zeros(0, numel(cfg.lb));
Fall = zeros(0, 2);
CVall = zeros(0, 1);

for gen = 0:G

    Xnew = init_pop(P, cfg.lb, cfg.ub);
    [Fnew, CVnew] = eval_pop(prob, Xnew);

    Xall = [Xall; Xnew];
    Fall = [Fall; Fnew];
    CVall = [CVall; CVnew];

    % select elite population of size P from archive (same selection as NSGA-II)
    [rank,crowd] = nsga2_rank_crowd(Fall, CVall);
    [~,idx] = sortrows([rank,-crowd],[1 2]);
    idx = idx(1:min(P,numel(idx)));

    X = Xall(idx,:);
    F = Fall(idx,:);
    CV = CVall(idx);

    hist(gen+1) = record_history(gen, prob, X, F, CV);

    if cfg.print.verbose && mod(gen, cfg.print.gen_print_every)==0
        feasN = sum(CV<=0);
        fprintf('[RANDOM gen %3d/%3d] feas=%3d/%3d | archive=%d\n', ...
            gen, G, feasN, size(X,1), size(Xall,1));
    end
end
end


%% =====================================================================
%%                       POP + MUTATION OPERATORS
%% =====================================================================

function X=init_pop(P,lb,ub)
n=numel(lb); X=zeros(P,n);
for i=1:P
    xi = lb + rand(1,n).*(ub-lb);
    xi(1:3)=round(xi(1:3));
    X(i,:)=xi;
end
end

function [F,CV]=eval_pop(prob,X)
P=size(X,1);
F=zeros(P,2); CV=zeros(P,1);
for i=1:P
    [fi,cvi] = eval_candidate(X(i,:),prob);
    F(i,:)=fi; CV(i)=cvi;
end
end

function [c1,c2]=sbx_mixed(p1,p2,lb,ub,eta_c)
c1=p1; c2=p2;
for j=1:3
    if rand<0.5, c1(j)=p2(j); c2(j)=p1(j); end
end
for j=4:numel(p1)
    u=rand;
    if u<=0.5, beta=(2*u)^(1/(eta_c+1));
    else, beta=(1/(2*(1-u)))^(1/(eta_c+1));
    end
    c1(j)=0.5*((1+beta)*p1(j)+(1-beta)*p2(j));
    c2(j)=0.5*((1-beta)*p1(j)+(1+beta)*p2(j));
end
c1=min(max(c1,lb),ub); c2=min(max(c2,lb),ub);
c1(1:3)=round(c1(1:3)); c2(1:3)=round(c2(1:3));
end

function x=poly_mut_mixed(x,lb,ub,pm,eta_m)
n=numel(x);
for j = 1:3
    if rand < pm
        lo = round(lb(j));
        hi = round(ub(j));
        if ~isfinite(lo) || ~isfinite(hi) || lo > hi
            error('Invalid integer bounds in poly_mut_mixed at j=%d: [%g, %g]', j, lb(j), ub(j));
        end
        x(j) = randi([lo, hi]);
    end
end
for j=4:n
    if rand<pm
        u=rand;
        if u<0.5, delta=(2*u)^(1/(eta_m+1))-1;
        else, delta=1-(2*(1-u))^(1/(eta_m+1));
        end
        x(j)=x(j)+delta*(ub(j)-lb(j));
        x(j)=min(max(x(j),lb(j)),ub(j));
    end
end
x(1:3)=round(x(1:3));
x(1:3)=min(max(x(1:3),lb(1:3)),ub(1:3));
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function [basis, ra] = build_deepc_basis(W, sig_thr, prob, cfg, Tini, N)
m = prob.m; p = prob.p;
K = Tini + N;
fprintf('DEBUG cfg.ro.enable=%d, K=%d, size(W)=[%d %d]\n', ...
    cfg.ro.enable, K, size(W,1), size(W,2));

if cfg.ro.enable
    [Uu,Ss,~] = svd(W,'econ');
    s = diag(Ss);

    ra = find(s >= sig_thr, 1, 'last');
    if isempty(ra), ra = 1; end

    if cfg.ro.ra_min_floor
        % ra_min = m*K + prob.n;
        if isfield(cfg.ro,'ra_min') && ~isempty(cfg.ro.ra_min)
            ra_min = cfg.ro.ra_min;
        else
            ra_min = m*Tini + prob.n;
        end
        ra = max(ra, min(ra_min, size(Uu,2)));
    end
    ra = min(ra, size(Uu,2));

    M00 = Uu(:,1:ra) * diag(s(1:ra));
    U00 = M00(1:m*K,:);
    Y00 = M00(m*K+1:end,:);
else
    ra  = size(W,2);
    U00 = W(1:m*K,:);
    Y00 = W(m*K+1:end,:);
end

basis.Up = U00(1:m*Tini,:);
basis.Uf = U00(m*Tini+1:end,:);
basis.Yp = Y00(1:p*Tini,:);
basis.Yf = Y00(p*Tini+1:end,:);
end

function kkt = factorize_deepc_kkt(Up,Uf,Yp,Yf, lam_g, lam_sig, m,p,Tini,N, cfg)

if nargin < 11 || isempty(cfg), cfg = struct(); end

ra = size(Up,2);
ps = p*Tini;

% ---- Build Q (output tracking) ----
Qy = 50;
if isfield(cfg,'deepc') && isfield(cfg.deepc,'Qy') && ~isempty(cfg.deepc.Qy)
    Qy = cfg.deepc.Qy;
end
Q1 = local_make_weight_matrix(Qy, p);
Q  = kron(eye(N), Q1);

% ---- Build R (input penalty) ----
Ru = [];
if isfield(cfg,'deepc') && isfield(cfg.deepc,'Ru') && ~isempty(cfg.deepc.Ru)
    Ru = cfg.deepc.Ru;
end
if isempty(Ru)
    % fallback
    alpha = 1e-2;
    if isfield(cfg,'deepc') && isfield(cfg.deepc,'alpha') && ~isempty(cfg.deepc.alpha)
        alpha = cfg.deepc.alpha;
    end
    R1 = alpha*eye(m);
else
    R1 = local_make_weight_matrix(Ru, m);
end
R = kron(eye(N), R1);

beta_du = 0;
if isfield(cfg,'deepc') && isfield(cfg.deepc,'beta_du') && ~isempty(cfg.deepc.beta_du)
    beta_du = cfg.deepc.beta_du;
end

Uf_first = Uf(1:m,:);
Uf1TUf1  = (Uf_first' * Uf_first);

Aeq = [Up, zeros(m*Tini, ps);
    Yp, eye(ps)];

% Hessian blocks
H11 = 2*(Yf'*Q*Yf + Uf'*R*Uf + lam_g*eye(ra) + beta_du*Uf1TUf1);
H22 = 2*(lam_sig*eye(ps));
H   = blkdiag(H11, H22);

KKT = [H,   Aeq';
    Aeq, zeros(size(Aeq,1))];

% jitter settings
if ~isfield(cfg,'con') || isempty(cfg.con), cfg.con = struct(); end
if ~isfield(cfg.con,'rcondKKT_min'), cfg.con.rcondKKT_min = 1e-12; end
if ~isfield(cfg.con,'kkt_jitter0'), cfg.con.kkt_jitter0 = 1e-10; end
if ~isfield(cfg.con,'kkt_jitter_growth'), cfg.con.kkt_jitter_growth = 10; end
if ~isfield(cfg.con,'kkt_jitter_max'), cfg.con.kkt_jitter_max = 1e-2; end
if ~isfield(cfg.con,'ldl_pivot_min'), cfg.con.ldl_pivot_min = 1e-14; end

% try LDL with adaptive jitter
jitter = cfg.con.kkt_jitter0;
best = [];
best_rc = -inf;

maxTries = 10;
for it=1:maxTries
    Kj = KKT + jitter*eye(size(KKT));
    rc = rcond(Kj); if ~isfinite(rc), rc = 0; end

    try
        [L,D,perm] = ldl(Kj,'vector');
        piv = abs(diag(D));
        pmin = min(piv(isfinite(piv)));
        if isempty(pmin), pmin = 0; end
        if pmin < cfg.con.ldl_pivot_min
            error('LDL:PivotsTooSmall','LDL pivots too small');
        end

        best = struct('L',L,'D',D,'perm',perm,'rc',rc,'jitter',jitter);
        best_rc = rc;
        if rc >= cfg.con.rcondKKT_min
            break;
        end
    catch
        % increase jitter
    end

    jitter = min(jitter*cfg.con.kkt_jitter_growth, cfg.con.kkt_jitter_max);
end

if isempty(best)
    Kj = KKT + cfg.con.kkt_jitter_max*eye(size(KKT));
    [L,D,perm] = ldl(Kj,'vector');
    best = struct('L',L,'D',D,'perm',perm,'rc',rcond(Kj),'jitter',cfg.con.kkt_jitter_max);
    best_rc = best.rc;
end

kkt = struct();
kkt.nx = ra + ps;
kkt.ra = ra;


kkt.Uf_first = Uf_first;
kkt.Uf_first_T = Uf_first';
kkt.beta_du = beta_du;

kkt.rhs_top = zeros(kkt.nx,1);

kkt.YfTQ = (Yf'*Q);

kkt.L = best.L;
kkt.D = best.D;
kkt.perm = best.perm;

kkt.rcondKKT = best_rc;
kkt.jitter = best.jitter;
end

function W = local_make_weight_matrix(val, dim)

if isscalar(val)
    W = val*eye(dim);
elseif isvector(val)
    v = val(:);
    if numel(v) ~= dim, v = v(1)*ones(dim,1); end
    W = diag(v);
else
    if ~isequal(size(val),[dim dim])
        W = val(1,1)*eye(dim);
    else
        W = val;
    end
end
end


function [u0, ok] = solve_deepc_kkt_cached(kkt, beq)


rhs = [kkt.rhs_top; beq];

% locally silence singular warnings
w1 = warning('query','MATLAB:nearlySingularMatrix');
w2 = warning('query','MATLAB:singularMatrix');
w3 = warning('query','MATLAB:illConditionedMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

try
    rp = rhs(kkt.perm);
    y  = kkt.L \ rp;
    z  = kkt.D \ y;
    w  = kkt.L' \ z;

    sol = zeros(size(rhs));
    sol(kkt.perm) = w;

    x = sol(1:kkt.nx);
    g = x(1:kkt.ra);

    u0 = kkt.Uf_first * g;
    ok = all(isfinite(u0));
catch
    u0 = nan(size(kkt.Uf_first,1),1);
    ok = false;
end

% restore warning states
warning(w1.state,'MATLAB:nearlySingularMatrix');
warning(w2.state,'MATLAB:singularMatrix');
warning(w3.state,'MATLAB:illConditionedMatrix');
end


function cfg = data_driven_bounds_from_offline(cfg, prob)
% Data-driven bounds using offline data scale + RO feasibility

m = prob.m; p = prob.p; n = prob.n;
Tfull = prob.Tfull;

% --- pick compute-feasible K range (still data-compatible) ---
Tini_lb = max(cfg.lb(1), n+6);
Tini_ub = min(cfg.ub(1), 32);
N_lb    = max(cfg.lb(2), 10);
N_ub    = min(cfg.ub(2), 50);

% implied K range
Kmax = Tini_ub + N_ub;
Kmin = Tini_lb + N_lb;

% --- Delta must support RO floor: Delta >= m*K + n - 1 + margin ---
margin = 30;
Delta_lb = max(cfg.lb(3), m*Kmax + n - 1 + margin);

% dataset length limit: K + Delta <= Tfull (must hold)
Delta_ub = min(cfg.ub(3), Tfull - Kmin);

if Delta_lb > Delta_ub
    Delta_lb = max(cfg.lb(3), m*Kmin + n - 1 + margin);
    Delta_ub = min(cfg.ub(3), Tfull - Kmin);
end

% --- Estimate a typical singular value scale to set sig_thr bounds ---

Tini0 = round((Tini_lb + Tini_ub)/2);
N0    = round((N_lb + N_ub)/2);
K0    = Tini0 + N0;
Delta0 = round(min(max(Delta_lb, m*K0+n-1+margin), Delta_ub));
Tdata0 = K0 + Delta0;

Tsrc1 = size(prob.Udata_full,2);
s01 = 1 + floor((Tsrc1 - Tdata0)/2);
s01 = max(1, min(s01, Tsrc1 - Tdata0 + 1));
U = prob.Udata_full(:, s01:s01+Tdata0-1);
Y = prob.Ydata_full(:, s01:s01+Tdata0-1);
Hu = build_hankel(U, K0);
Hy = build_hankel(Y, K0);
W0 = [Hu; Hy];

sW = svd(W0,'econ');
smed = median(sW);

sig_thr_low  = max(cfg.con.sig_thr_min, 0.05*smed);
sig_thr_high = max(sig_thr_low*10,      0.50*smed); % ensure separation

% --- Tighten log10(sig_thr) bounds to data scale ---
cfg.lb(6) = log10(sig_thr_low);
cfg.ub(6) = log10(sig_thr_high);

% --- Tighten Delta bounds ---
cfg.lb(3) = Delta_lb;
cfg.ub(3) = Delta_ub;

% --- Tighten Tini/N bounds (optional but helps robustness/compute) ---
cfg.lb(1) = Tini_lb; cfg.ub(1) = Tini_ub;
cfg.lb(2) = N_lb;    cfg.ub(2) = N_ub;

% --- Practical ridge bounds (stop extreme lam_sig explosions) ---
% Keep wide-ish, but not insane:
cfg.lb(4) = max(cfg.lb(4), -6);  cfg.ub(4) = min(cfg.ub(4),  2); % lam_g in [1e-6, 1e2]
cfg.lb(5) = max(cfg.lb(5), -6);  cfg.ub(5) = min(cfg.ub(5),  3); % lam_sig in [1e-6, 1e3]
end

function cfg = data_driven_bounds_from_pilot(cfg, prob)

% -----------------------------
% settings and defaults
% -----------------------------
if ~isfield(cfg,'bounds') || isempty(cfg.bounds)
    error('cfg.bounds must exist for pilot_data_driven mode.');
end

if ~isfield(cfg.bounds,'super_lb') || isempty(cfg.bounds.super_lb)
    super_lb = cfg.lb;
else
    super_lb = cfg.bounds.super_lb;
end

if ~isfield(cfg.bounds,'super_ub') || isempty(cfg.bounds.super_ub)
    super_ub = cfg.ub;
else
    super_ub = cfg.bounds.super_ub;
end

if ~isfield(cfg.bounds,'nPilot') || isempty(cfg.bounds.nPilot)
    nPilot = 300;
else
    nPilot = cfg.bounds.nPilot;
end

if ~isfield(cfg.bounds,'keepFrac') || isempty(cfg.bounds.keepFrac)
    keepFrac = 0.20;
else
    keepFrac = cfg.bounds.keepFrac;
end

if ~isfield(cfg.bounds,'padFrac') || isempty(cfg.bounds.padFrac)
    padFrac = 0.10;
else
    padFrac = cfg.bounds.padFrac;
end

if ~isfield(cfg.bounds,'minKeep') || isempty(cfg.bounds.minKeep)
    minKeep = 20;
else
    minKeep = cfg.bounds.minKeep;
end

nvar = numel(cfg.lb);

% -----------------------------
% sample pilot population in super-box
% -----------------------------
Xp = zeros(nPilot, nvar);
for i = 1:nPilot
    xi = super_lb + rand(1,nvar).*(super_ub - super_lb);
    xi(1:3) = round(xi(1:3));
    xi = min(max(xi, super_lb), super_ub);
    xi(1:3) = round(xi(1:3));
    Xp(i,:) = xi;
end

cfgPilot = cfg;
cfgPilot.cache.enable = false;

if ~isfield(cfgPilot,'con') || isempty(cfgPilot.con)
    cfgPilot.con = struct();
end
cfgPilot.con.use_tail_constraint = false;

probPilot = prob;
probPilot.cfg = cfgPilot;
probPilot.cache = [];

Fpilot  = nan(nPilot, 2);
CVpilot = nan(nPilot, 1);

for i = 1:nPilot
    [fi, cvi] = eval_candidate(Xp(i,:), probPilot, 'calibration_mode', true);
    Fpilot(i,:)  = fi;
    CVpilot(i,1) = cvi;
end

feas = (CVpilot <= 0);
nKeep = max(minKeep, ceil(keepFrac * nPilot));
nKeep = min(nKeep, nPilot);

if sum(feas) >= minKeep
    Xf = Xp(feas,:);
    Ff = Fpilot(feas,:);

    % normalize objectives for scalar screening
    fmin = min(Ff,[],1);
    fmax = max(Ff,[],1);
    den  = max(fmax - fmin, 1e-12);

    score = sum((Ff - fmin)./den, 2);
    [~,ord] = sort(score, 'ascend');

    nkeep_feas = min(numel(ord), nKeep);
    Xkeep = Xf(ord(1:nkeep_feas), :);

else
    [~,ord] = sort(CVpilot, 'ascend');
    Xkeep = Xp(ord(1:nKeep), :);
end

if isempty(Xkeep)
    warning('Pilot tightening failed: no kept points. Using current bounds.');
    return;
end

new_lb = zeros(1,nvar);
new_ub = zeros(1,nvar);

for j = 1:nvar
    xj = Xkeep(:,j);
    xj = xj(isfinite(xj));

    if isempty(xj)
        new_lb(j) = cfg.lb(j);
        new_ub(j) = cfg.ub(j);
        continue;
    end

    qlo = local_quantile(xj, 0.10);
    qhi = local_quantile(xj, 0.90);

    span = qhi - qlo;
    if span < 1e-12
        span = max(1e-6, 0.05*(super_ub(j)-super_lb(j)));
    end

    lo = qlo - padFrac*span;
    hi = qhi + padFrac*span;

    lo = max(lo, super_lb(j));
    hi = min(hi, super_ub(j));

    % integer dimensions
    if j <= 3
        lo = round(lo);
        hi = round(hi);

        % keep at least width 1 if possible
        if hi < lo
            hi = lo;
        end
    end

    new_lb(j) = lo;
    new_ub(j) = hi;
end

% ensure Tdata = Tini + N + Delta remains possible within Tfull
Tfull = prob.Tfull;

% keep integer vars integral and valid
new_lb(1:3) = round(new_lb(1:3));
new_ub(1:3) = round(new_ub(1:3));


Kmin = new_lb(1) + new_lb(2);
Kmax = new_ub(1) + new_ub(2);

new_ub(3) = min(new_ub(3), Tfull - Kmin);
if new_ub(3) < new_lb(3)
    new_lb(3) = min(new_lb(3), Tfull - Kmin);
    new_ub(3) = max(new_lb(3), Tfull - Kmin);
end


m = prob.m; n = prob.n;
margin = 30;
Delta_floor = m*Kmax + n - 1 + margin;
new_lb(3) = max(new_lb(3), Delta_floor);

if new_lb(3) > new_ub(3)
    new_lb(3) = min(new_lb(3), new_ub(3));
end

% final clipping to the original default box is optional:
% here we keep the pilot box, not the old manual box.

cfg.lb = new_lb;
cfg.ub = new_ub;

cfg.bounds.lb_pilot = new_lb;
cfg.bounds.ub_pilot = new_ub;
cfg.bounds.Xpilot   = Xp;
cfg.bounds.Fpilot   = Fpilot;
cfg.bounds.CVpilot  = CVpilot;
cfg.bounds.Xkeep    = Xkeep;

fprintf('Pilot-tightened bounds:\n');
fprintf('  lb = [%g %g %g %g %g %g]\n', cfg.lb);
fprintf('  ub = [%g %g %g %g %g %g]\n', cfg.ub);

end

function q = local_quantile(x, p)
% simple quantile helper, base MATLAB
x = sort(x(:));
n = numel(x);

if n == 0
    q = NaN;
    return;
elseif n == 1
    q = x(1);
    return;
end

pos = 1 + (n-1)*p;
lo  = floor(pos);
hi  = ceil(pos);

if lo == hi
    q = x(lo);
else
    w = pos - lo;
    q = (1-w)*x(lo) + w*x(hi);
end
end

function hv = hv2d_min(F, ref)
hv = 0;
if isempty(F), return; end

F = F(all(isfinite(F),2),:);
if isempty(F), return; end

% keep only points that dominate (are <=) the reference
keep = (F(:,1) <= ref(1)) & (F(:,2) <= ref(2));
F = F(keep,:);
if isempty(F), return; end

% sort by f1 ascending
F = sortrows(F, 1);

% sweep: add rectangles when f2 improves
prev_f2 = ref(2);
best_f2 = inf;
for i=1:size(F,1)
    f1 = F(i,1);
    f2 = F(i,2);

    if f2 < best_f2
        % skyline improvement
        hv = hv + max(0, (ref(1) - f1)) * max(0, (prev_f2 - f2));
        prev_f2 = f2;
        best_f2 = f2;
    end
end
end

function sp = spacing_metric(F)

n = size(F,1);
if n < 2
    sp = NaN;
    return;
end

d = zeros(n,1);
for i=1:n
    best = inf;
    for j=1:n
        if i==j, continue; end
        dij = norm(F(i,:) - F(j,:), 1);
        if dij < best, best = dij; end
    end
    d(i) = best;
end

dm = mean(d);
sp = sqrt(sum((d - dm).^2) / max(1,(n-1)));
end

function spr = max_spread(F, ref)

if isempty(F)
    spr = 0; return;
end

F = F(all(isfinite(F),2),:);
if isempty(F)
    spr = 0; return;
end

ideal = min(F,[],1);
rngv  = max(F,[],1) - ideal;

den = max(ref - ideal, eps);
spr = mean(min(1, rngv ./ den));
end


function roll = deepc_rollout_api(prob, xbest, opts)

cfg = prob.cfg;

Tini  = round(xbest(1));
N     = round(xbest(2));
Delta = round(xbest(3));

lam_g   = max(10^(xbest(4)), cfg.con.lam_g_min);
lam_sig = max(10^(xbest(5)), cfg.con.lam_sig_min);
sig_thr = max(10^(xbest(6)), cfg.con.sig_thr_min);

K = Tini + N;
Tdata = K + Delta;

Tsrc2 = size(prob.Udata_full,2);

if Tdata > Tsrc2
    Tdata = Tsrc2;
end

s02 = 1 + floor((Tsrc2 - Tdata)/2);
s02 = max(1, min(s02, Tsrc2 - Tdata + 1));

U = prob.Udata_full(:, s02:s02+Tdata-1);
Y = prob.Ydata_full(:, s02:s02+Tdata-1);
Hu = build_hankel(U, K);
Hy = build_hankel(Y, K);
W0 = [Hu; Hy];

% --- make a local copy so we don’t mutate caller state ---
prob2 = prob;
cfg2  = prob2.cfg;

% override toggles
if isfield(opts,'ro_enable'),      cfg2.ro.enable = logical(opts.ro_enable); end
if isfield(opts,'online_mode'),    cfg2.online.update_mode = opts.online_mode; end
if isfield(opts,'update_period'),  cfg2.online.update_period = opts.update_period; end
if isfield(opts,'Ksim'),           prob2.Ksim = opts.Ksim; end
if isfield(opts,'window_max_cols'), cfg2.online.window_max_cols = opts.window_max_cols; end

prob2.cfg = cfg2;

if isfield(prob,'ref_fun')
    prob2.ref_fun = prob.ref_fun;
else
    prob2.ref_fun = [];
end

if isfield(prob,'ref') && ~isempty(prob.ref)
    prob2.ref = deepc_extend_ref(prob.ref, prob2.p, prob2.Ksim, prob2.ref_fun);
else
    prob2.ref = deepc_extend_ref([], prob2.p, prob2.Ksim, prob2.ref_fun);
end

% ---- Ensure reference trajectory exists for the rollout horizon ----
if isfield(opts,'seed'), rng(opts.seed); end
x0 = prob2.x0_nom;
lambda_shift = 0;
met = simulate_closed_loop(prob2, Tini, N, W0, lam_g, lam_sig, sig_thr, x0, lambda_shift, true);

roll = struct();
roll.u_log      = met.u_log;
roll.y_log      = met.y_log;
roll.lam_log    = met.lam_log;
roll.ra_log     = met.ra_log;
roll.L_log      = met.L_log;
roll.ro_enable_used = cfg2.ro.enable;
roll.t_solve    = met.t_solve;
roll.t_rebuild  = met.t_rebuild;

roll.r_log = met.r_log;

% summary
roll.ra_final = met.ra_log(find(~isnan(met.ra_log),1,'last'));
roll.L_final  = met.L_log(find(~isnan(met.L_log),1,'last'));
end




function etmax = calibrate_e_tail_max(prob, cfg, nCal, qTail, slack)

if nargin < 3 || isempty(nCal),  nCal  = 200; end
if nargin < 4 || isempty(qTail), qTail = 0.90; end
if nargin < 5 || isempty(slack), slack = 0.10; end

X = init_pop(nCal, cfg.lb, cfg.ub);
et = nan(nCal,1);

for i = 1:nCal
    x = X(i,:);
    [~,~,det] = deepc_eval(x, prob);

    if isfield(det,'e_tail_agg') && isfinite(det.e_tail_agg)
        et(i) = det.e_tail_agg;
    end
end

et = et(isfinite(et));

if isempty(et)
    error('Calibration failed: no finite e_tail_agg values collected.');
end

fprintf('CAL tail stats: n=%d min=%.4g med=%.4g p90=%.4g p95=%.4g max=%.4g\n', ...
    numel(et), min(et), median(et), prctile(et,90), prctile(et,95), max(et));

etmax = prctile(et, 100*qTail) * (1 + slack);
end

function print_run_header(cfg)
fprintf('============================================================\n');
fprintf('Algo: %s | P=%d | G=%d | Scenarios S=%d | q=%.2f\n', ...
    upper(cfg.algo), cfg.P, cfg.G, cfg.obj.use_robust*cfg.obj.S + (~cfg.obj.use_robust)*1, cfg.obj.q);
fprintf('Decision x=[Tini N Delta log10(lg) log10(ls) log10(sig_thr)]\n');
fprintf('Objectives: Jctrl=mean(IAE)+alpha*CVaR(IAE)+beta_u*mean(U2)+beta_v*mean(viol)\n');
fprintf('            Jtime=%s(t)+gamma*CVaR(t)\n', cfg.obj.time_stat);
fprintf('Best-per-generation printing: %s (every %d gen)\n', cfg.print.best_choice, cfg.print.gen_print_every);
fprintf('RO: %d | OnlineUpdate: %s (period=%d) | Cache: %d\n', ...
    cfg.ro.enable, cfg.online.update_mode, cfg.online.update_period, cfg.cache.enable);
fprintf('============================================================\n');
end
