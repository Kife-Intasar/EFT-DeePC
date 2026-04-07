function nd = deepc_nondominated_mask(F)
N = size(F,1);
nd = true(N,1);

bad = ~all(isfinite(F),2);
nd(bad) = false;

for i = 1:N
    if ~nd(i), continue; end
    for j = 1:N
        if i==j || ~nd(j), continue; end
        if all(F(j,:)<=F(i,:)) && any(F(j,:)<F(i,:))
            nd(i)=false; break;
        end
    end
end
end
