function [Udata,Ydata] = deepc_generate_offline_data(prob, T, lambda_shift)

m = prob.m; p = prob.p; n = prob.n;

umin = prob.umin(:); umax = prob.umax(:);
if numel(umin)==1, umin = umin*ones(m,1); end
if numel(umax)==1, umax = umax*ones(m,1); end

Udata = zeros(m,T);
Ydata = zeros(p,T);

x = prob.x0_nom;

umid = 0.5*(umin+umax);
uamp = 0.5*(umax-umin);

% switching control
hold = 0;
u = zeros(m,1);

for k=1:T
    if hold<=0
        hold = randi([1 3]);
        sgn = sign(randn(m,1));  % ±1
        sgn(sgn==0) = 1;
        u = umid + uamp .* sgn;
        u = u + 0.10*uamp .* (2*rand(m,1)-1);
        % enforce bounds
        u = min(max(u, umin), umax);
    end
    hold = hold - 1;

    Udata(:,k) = u;

    y = prob.C*x + deepc_bounded_noise(prob.dm_max,p);
    Ydata(:,k) = y;

    [A,B] = deepc_AB_of_k(prob,k,lambda_shift);
    x = A*x + B*u + deepc_bounded_noise(prob.dp_max,n);
end
end
