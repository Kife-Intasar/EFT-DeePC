function ref = deepc_hv_ref_from_points(F, padFrac)
if nargin < 2 || isempty(padFrac), padFrac = 0.10; end

if isempty(F)
    ref = [1 1];
    return;
end

F = F(all(isfinite(F),2), :);
if isempty(F)
    ref = [1 1];
    return;
end

mx  = max(F, [], 1);

pad = padFrac * max(1, abs(mx));
ref = mx + pad;

ref = ref + 1e-12;
end
