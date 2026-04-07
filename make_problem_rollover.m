function prob = make_problem_rollover(cfg)

prob.n=4; prob.m=1; prob.p=1;

Ts = 0.1;
if isfield(cfg,'prob') && isfield(cfg.prob,'Ts') && ~isempty(cfg.prob.Ts)
    Ts = cfg.prob.Ts;
end

Ac = [ 0.00499   0.997    0.0154   -6.81e-5;
      -78.3    -12.2    -65.3     -3.89;
       -0.932   -0.799   -6.20    -1.57;
        1.52     3.32     8.27    -1.49 ];

Bc = [-5.76e-5; 2.80; 0.278; 0.655];

prob.A0 = eye(4) + Ts*Ac;
prob.B0 = Ts*Bc;

prob.C  = [0.1200 0.0124 -0.0108 0.0109];

prob.Ad = zeros(size(prob.A0));
prob.Bd = zeros(size(prob.B0));

prob.dp_max = 0.001;
prob.dm_max = 0.001;

prob.x0_nom = [0.05; 0; 0.5; 0];

prob.umin = -90;
prob.umax =  90;

prob.tol_rank = 1e-12;
prob.Ksim  = cfg.prob.Ksim;
prob.Tfull = cfg.prob.Tfull;
prob.y_max = 1.0;
cfg.prob.ref_val = 0;
cfg.prob.ref_vec = zeros(prob.p,1);
cfg.prob.ref_fun = @(t,p) zeros(p,1);   

[cfg, prob] = deepc_finalize_reference(cfg, prob);

st = rng; rng(cfg.seed_data);
[U,Y] = deepc_generate_offline_data(prob, prob.Tfull, 0);
rng(st);

prob.Udata_full = U;
prob.Ydata_full = Y;

prob.kappa_max = cfg.con.kappa_max;
prob.sigma_min = cfg.con.sigma_min;
prob.fail_rate_max = cfg.con.fail_rate_max;
prob.x_max = cfg.con.x_max;
prob.e_tail_max = cfg.con.e_tail_max;

prob.beta_u = cfg.deepc.beta_u;
prob.beta_v = cfg.deepc.beta_v;
prob.beta_y = cfg.deepc.beta_y;

prob.cfg = cfg;
end
