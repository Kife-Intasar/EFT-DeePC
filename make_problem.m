function prob = make_problem(cfg)
% Factory: chooses plant by cfg.prob.model

model = 'ltv';
if isfield(cfg,'prob') && isfield(cfg.prob,'model') && ~isempty(cfg.prob.model)
    model = lower(cfg.prob.model);
end

switch model
    % case 'ltv'
    %     prob = make_problem_ltv(cfg);
    case 'rollover'
        prob = make_problem_rollover(cfg);
    % case 'quadruple_tank'
    %     prob = make_problem_quadruple_tank(cfg);
    % case 'lfc'
    %     prob = make_problem_lfc(cfg);
    otherwise
        error('Unknown cfg.prob.model = %s', model);
end

% ---- set HV reference deterministically (same across algorithms) ----
cfg = deepc_set_hv_ref(cfg, prob);
prob.cfg = cfg;   % IMPORTANT: write back so deepc_run uses it



end
