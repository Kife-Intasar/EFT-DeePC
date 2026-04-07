function T = summarize_suite(suite, padFrac)

if nargin < 2 || isempty(padFrac), padFrac = 0.10; end

[nCfg, R] = size(suite.runs);

% Fixed global HV reference
ref = deepc_hv_ref_from_suite(suite, padFrac);

names = string(suite.names(:));

HV        = nan(nCfg, R);
Jctrl     = nan(nCfg, R);
Jtime     = nan(nCfg, R);
FeasRat   = nan(nCfg, R);
NDcount   = nan(nCfg, R);

TailCtrl  = nan(nCfg, R);   
TailTime  = nan(nCfg, R);   
CVTrCtrl  = nan(nCfg, R);   
CVTrTime  = nan(nCfg, R);  

for i = 1:nCfg
    for r = 1:R
        out = suite.runs{i,r};
        if isempty(out), continue; end

        h = out.hist(end);

        % ---- final HV from final feasible ND points ----
        Fnd = h.Fnd;
        HV(i,r) = deepc_hv2d(Fnd, ref);

        % ---- final feasible ratio + ND count ----
        FeasRat(i,r) = h.feasible_ratio;
        NDcount(i,r) = h.nd_count;

        % ---- best feasible ctrl/time objective values ----
        if isfield(h,'bestCtrlFeas') && all(isfinite(h.bestCtrlFeas))
            Jctrl(i,r) = h.bestCtrlFeas(1);
        else
            Jctrl(i,r) = local_best_feas(out.final.F, out.final.CV, 1);
        end

        if isfield(h,'bestTimeFeas') && all(isfinite(h.bestTimeFeas))
            Jtime(i,r) = h.bestTimeFeas(2);
        else
            Jtime(i,r) = local_best_feas(out.final.F, out.final.CV, 2);
        end

        % ---- tail metrics ----
        if isfield(h,'bestCtrlTail') && isfinite(h.bestCtrlTail)
            TailCtrl(i,r) = h.bestCtrlTail;
        elseif isfield(h,'bestCtrlX') && all(isfinite(h.bestCtrlX))
            oldCache = out.prob.cfg.cache.enable;
            out.prob.cfg.cache.enable = false;
            [~,~,detC] = deepc_eval(h.bestCtrlX, out.prob);
            out.prob.cfg.cache.enable = oldCache;

            if isfield(detC,'e_tail_agg'), TailCtrl(i,r) = detC.e_tail_agg; end
            if isfield(detC,'cv_track'),   CVTrCtrl(i,r) = detC.cv_track;   end
        end

        if isfield(h,'bestTimeTail') && isfinite(h.bestTimeTail)
            TailTime(i,r) = h.bestTimeTail;
        elseif isfield(h,'bestTimeX') && all(isfinite(h.bestTimeX))
            oldCache = out.prob.cfg.cache.enable;
            out.prob.cfg.cache.enable = false;
            [~,~,detT] = deepc_eval(h.bestTimeX, out.prob);
            out.prob.cfg.cache.enable = oldCache;

            if isfield(detT,'e_tail_agg'), TailTime(i,r) = detT.e_tail_agg; end
            if isfield(detT,'cv_track'),   CVTrTime(i,r) = detT.cv_track;   end
        end

        if isfield(h,'bestCtrlCVtrack') && isfinite(h.bestCtrlCVtrack)
            CVTrCtrl(i,r) = h.bestCtrlCVtrack;
        end

        if isfield(h,'bestTimeCVtrack') && isfinite(h.bestTimeCVtrack)
            CVTrTime(i,r) = h.bestTimeCVtrack;
        end
    end
end

% ---- stats ----
[HV_med, HV_iqr, HV_mu, HV_sd]             = row_stats(HV);
[Jc_med, Jc_iqr, Jc_mu, Jc_sd]             = row_stats(Jctrl);
[Jt_med, Jt_iqr, Jt_mu, Jt_sd]             = row_stats(Jtime);
[Fr_med, Fr_iqr, Fr_mu, Fr_sd]             = row_stats(FeasRat);
[Nd_med, Nd_iqr, Nd_mu, Nd_sd]             = row_stats(NDcount);
[Tc_med, Tc_iqr, Tc_mu, Tc_sd]             = row_stats(TailCtrl);
[Tt_med, Tt_iqr, Tt_mu, Tt_sd]             = row_stats(TailTime);
[CVTc_med, CVTc_iqr, CVTc_mu, CVTc_sd]     = row_stats(CVTrCtrl);
[CVTt_med, CVTt_iqr, CVTt_mu, CVTt_sd]     = row_stats(CVTrTime);

% ---- formatted strings ----
HV_fmt   = fmt_pm(HV_med, HV_iqr);
Jc_fmt   = fmt_pm(Jc_med, Jc_iqr);
Jt_fmt   = fmt_pm(Jt_med, Jt_iqr);
Fr_fmt   = fmt_pm(Fr_med, Fr_iqr);
Nd_fmt   = fmt_pm(Nd_med, Nd_iqr);
Tc_fmt   = fmt_pm(Tc_med, Tc_iqr);
Tt_fmt   = fmt_pm(Tt_med, Tt_iqr);
CVTc_fmt = fmt_pm(CVTc_med, CVTc_iqr);
CVTt_fmt = fmt_pm(CVTt_med, CVTt_iqr);

T = table( ...
    names, ...
    HV_med, HV_iqr, HV_mu, HV_sd, HV_fmt, ...
    Jc_med, Jc_iqr, Jc_mu, Jc_sd, Jc_fmt, ...
    Jt_med, Jt_iqr, Jt_mu, Jt_sd, Jt_fmt, ...
    Fr_med, Fr_iqr, Fr_mu, Fr_sd, Fr_fmt, ...
    Nd_med, Nd_iqr, Nd_mu, Nd_sd, Nd_fmt, ...
    Tc_med, Tc_iqr, Tc_mu, Tc_sd, Tc_fmt, ...
    Tt_med, Tt_iqr, Tt_mu, Tt_sd, Tt_fmt, ...
    CVTc_med, CVTc_iqr, CVTc_mu, CVTc_sd, CVTc_fmt, ...
    CVTt_med, CVTt_iqr, CVTt_mu, CVTt_sd, CVTt_fmt, ...
    'VariableNames', { ...
      'name', ...
      'HV_median','HV_IQR','HV_mean','HV_std','HV_median_pm_IQR', ...
      'Jctrl_median','Jctrl_IQR','Jctrl_mean','Jctrl_std','Jctrl_median_pm_IQR', ...
      'Jtime_median','Jtime_IQR','Jtime_mean','Jtime_std','Jtime_median_pm_IQR', ...
      'feasRatio_median','feasRatio_IQR','feasRatio_mean','feasRatio_std','feasRatio_median_pm_IQR', ...
      'ndCount_median','ndCount_IQR','ndCount_mean','ndCount_std','ndCount_median_pm_IQR', ...
      'tailCtrl_median','tailCtrl_IQR','tailCtrl_mean','tailCtrl_std','tailCtrl_median_pm_IQR', ...
      'tailTime_median','tailTime_IQR','tailTime_mean','tailTime_std','tailTime_median_pm_IQR', ...
      'cvTrackCtrl_median','cvTrackCtrl_IQR','cvTrackCtrl_mean','cvTrackCtrl_std','cvTrackCtrl_median_pm_IQR', ...
      'cvTrackTime_median','cvTrackTime_IQR','cvTrackTime_mean','cvTrackTime_std','cvTrackTime_median_pm_IQR' ...
    } ...
);
end

% ================= helpers =================

function best = local_best_feas(F, CV, whichObj)
feas = (CV <= 0);
if ~any(feas), best = NaN; return; end
Ff = F(feas,:);
best = min(Ff(:,whichObj));
end

function [medv, iqrV, mu, sd] = row_stats(M)
n = size(M,1);
medv = nan(n,1); iqrV = nan(n,1); mu = nan(n,1); sd = nan(n,1);
for i=1:n
    x = M(i,:); x = x(isfinite(x));
    if isempty(x), continue; end
    x = sort(x(:));
    medv(i) = quantile_simple(x, 0.50);
    q1      = quantile_simple(x, 0.25);
    q3      = quantile_simple(x, 0.75);
    iqrV(i) = q3 - q1;
    mu(i)   = mean(x);
    sd(i)   = std(x,0);
end
end

function q = quantile_simple(x, p)
n = numel(x);
if n==0, q = NaN; return; end
if n==1, q = x(1); return; end
pos = 1 + (n-1)*p;
lo  = floor(pos);
hi  = ceil(pos);
if lo==hi
    q = x(lo);
else
    w = pos - lo;
    q = (1-w)*x(lo) + w*x(hi);
end
end

function s = fmt_pm(medv, iqrV)
n = numel(medv);
s = strings(n,1);
for i=1:n
    if ~isfinite(medv(i))
        s(i) = "NaN";
    else
        s(i) = sprintf('%.4g ± %.4g', medv(i), iqrV(i));
    end
end
end
