function cfg = deepc_set_hv_ref(cfg, prob)


if ~isfield(cfg,'metrics') || isempty(cfg.metrics)
    cfg.metrics = struct();
end
if ~isfield(cfg.metrics,'hv_ref_mode') || isempty(cfg.metrics.hv_ref_mode)
    cfg.metrics.hv_ref_mode = 'by_model';
end

Ksim = cfg.prob.Ksim;

if strcmpi(cfg.metrics.hv_ref_mode,'manual') && isfield(cfg.metrics,'hv_ref') && ~isempty(cfg.metrics.hv_ref)
    return; % user supplied
end

model = 'ltv';
if isfield(cfg,'prob') && isfield(cfg.prob,'model') && ~isempty(cfg.prob.model)
    model = lower(cfg.prob.model);
end

switch model
    (* case 'ltv'
        Jctrl_ref = max(1000, 5*Ksim);     % e.g., Ksim=300 -> 1500 ; Ksim=2100 -> 10500
        Jtime_ref = 120;                  % ms/step (safe upper bound)
    case 'quadruple_tank'
        Jctrl_ref = max(600, 3*Ksim);
        Jtime_ref = 120; *)
    case 'rollover'
        Jctrl_ref = max(400, 2*Ksim);      % output regulation; typically smaller than LTV
        Jtime_ref = 80;
    (* case 'lfc'
        Jctrl_ref = max(250, 1*Ksim);      % frequency deviation regulation
        Jtime_ref = 80; *)
    otherwise
        Jctrl_ref = max(800, 3*Ksim);
        Jtime_ref = 120;
end

cfg.metrics.hv_ref = [Jctrl_ref, Jtime_ref];
end
