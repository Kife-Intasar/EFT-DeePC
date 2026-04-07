function ref = deepc_hv_ref_from_suite(suite, padFrac)
% Fixed HV ref from union of all FINAL feasible ND points across suite
if nargin < 2 || isempty(padFrac), padFrac = 0.10; end

[nCfg, R] = size(suite.runs);

buf = cell(nCfg*R, 1);
k = 0;

for i = 1:nCfg
    for r = 1:R
        Fnd = suite.runs{i,r}.hist(end).Fnd;
        if ~isempty(Fnd)
            Fnd = Fnd(all(isfinite(Fnd),2), :);
            if ~isempty(Fnd)
                k = k + 1;
                buf{k} = Fnd;
            end
        end
    end
end

if k == 0
    ref = [1 1];
    return;
end

allFinal = vertcat(buf{1:k});
ref = deepc_hv_ref_from_points(allFinal, padFrac);
end
