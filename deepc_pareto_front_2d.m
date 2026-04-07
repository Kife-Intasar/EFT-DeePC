function PF = deepc_pareto_front_2d(F)
if isempty(F), PF = F; return; end

F = F(all(isfinite(F),2), :);
if isempty(F), PF = zeros(0,2); return; end

nd = deepc_nondominated_mask(F);
PF = F(nd,:);

PF = unique(PF, 'rows');

PF = sortrows(PF, 1, 'ascend');

keep = true(size(PF,1),1);
best_f2 = inf;
for i = 1:size(PF,1)
    if PF(i,2) < best_f2
        best_f2 = PF(i,2);
    else
        keep(i) = false;
    end
end
PF = PF(keep,:);
end
