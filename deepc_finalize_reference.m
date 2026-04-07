function [cfg, prob] = deepc_finalize_reference(cfg, prob)

p = prob.p;
K = prob.Ksim;

% If user provided ref_vec, prefer it for building prob.ref
if isfield(cfg,'prob') && isfield(cfg.prob,'ref_vec') && ~isempty(cfg.prob.ref_vec)
    rv = cfg.prob.ref_vec(:);
    if numel(rv) < p, rv(end+1:p) = rv(end); end
    if numel(rv) > p, rv = rv(1:p); end
    prob.ref = repmat(rv, 1, K);

    cfg.prob.ref_fun = @(t,pp) rv(1:min(pp,numel(rv)));
    prob.ref_fun = cfg.prob.ref_fun;
    return;
end

% If ref_fun missing but ref_val exists, create constant ref_fun
if ~isfield(cfg,'prob'), cfg.prob = struct(); end
if (~isfield(cfg.prob,'ref_fun') || ~isa(cfg.prob.ref_fun,'function_handle'))
    if isfield(cfg.prob,'ref_val') && ~isempty(cfg.prob.ref_val)
        rv = cfg.prob.ref_val;
        cfg.prob.ref_fun = @(t,pp) rv*ones(pp,1);
    else
        cfg.prob.ref_fun = []; % means zero reference
    end
end

% Build stored reference
if isa(cfg.prob.ref_fun,'function_handle')
    prob.ref = deepc_extend_ref([], p, K, cfg.prob.ref_fun);
    prob.ref_fun = cfg.prob.ref_fun;
else
    rv = 0;
    if isfield(cfg.prob,'ref_val') && ~isempty(cfg.prob.ref_val), rv = cfg.prob.ref_val; end
    prob.ref = rv*ones(p,K);
    prob.ref_fun = [];
end
end
