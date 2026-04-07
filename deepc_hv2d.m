function hv = deepc_hv2d(F, ref)

if isempty(F), hv = 0; return; end


F = F(all(isfinite(F),2), :);
if isempty(F), hv = 0; return; end

F = F(F(:,1) < ref(1) & F(:,2) < ref(2), :);
if isempty(F), hv = 0; return; end

F = deepc_pareto_front_2d(F);
F = sortrows(F, 1, 'ascend');

hv = 0;
prev_f2 = ref(2);
for i = 1:size(F,1)
    w = ref(1) - F(i,1);
    h = prev_f2 - F(i,2);
    if w > 0 && h > 0
        hv = hv + w*h;
    end
    prev_f2 = min(prev_f2, F(i,2));
end
end
