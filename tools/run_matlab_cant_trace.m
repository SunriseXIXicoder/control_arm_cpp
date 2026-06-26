function run_matlab_cant_trace()
src = 'C:\Users\administered\Desktop\拔模SIMP';
addpath(src);

out = getenv('TRACE_OUT');
if isempty(out)
    out = fullfile(pwd, 'result', 'matlab_cant_trace.csv');
end
maxiter = str2double(getenv('TRACE_MAXITER'));
if isnan(maxiter) || maxiter <= 0
    maxiter = 5;
end

DL = 10; DW = 10; DH = 10;
proport = 2;
nelx = 10 * proport; nely = 10 * proport; nelz = 10 * proport;
dx = [DL/nelx; DW/nely; DH/nelz];
[Eulerx0,Eulery0,Eulerz0] = meshgrid(dx(1)*(1:nelx),dx(2)*(1:nely),dx(3)*(1:nelz));
a = 0.05; b = 0.95;
xf = (Eulerx0-dx(1)/2)./DL; xf = a + xf*(b-a);
yf = (Eulery0-dx(2)/2)./DW; yf = a + yf*(b-a);
zf = (Eulerz0-dx(3)/2)./DH; zf = a + zf*(b-a);
volfrac = 0.2;
penal = 1;
model = 'cant';
E0 = 1;
Emin = 1e-9;
nu = 0.3;

[freedofs,iK,jK,KE,F,U,edofMat,loadele] = initiMesh(nelx,nely,nelz,model,nu);
nelx_Pj = nelx; nely_Pj = nely; nelz_Pj = nelz;
[H,Hs] = Sentivity_Filter(nelx_Pj,nely_Pj,nelz_Pj,max(min([nelx_Pj,nely_Pj,nelz_Pj]/3,5)));
[LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var] = SS(nelx,nely,nelz,DL,DW,DH,nelx_Pj,nely_Pj,nelz_Pj);

x1 = zeros(nely_Pj*nelz_Pj,1);
x2 = ones(nely_Pj*nelz_Pj,1);
y1 = zeros(nelx_Pj*nelz_Pj,1);
y2 = ones(nelx_Pj*nelz_Pj,1);
z1 = zeros(nelx_Pj*nely_Pj,1);
z2 = ones(nelx_Pj*nely_Pj,1);
x = [x1;x2;y1;y2;z1;z2];
beta1 = 10;
beta2 = 2;
mma = MMAInit(nelx_Pj,nely_Pj,nelz_Pj);
B = 1:2*nelz_Pj*nely_Pj+2*nelz_Pj*nelx_Pj;
bb = x(B);

fid = fopen(out, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, ['iter,f0val,fval,volume,beta1,beta2,move,c_scale,x_change,' ...
    'x_min,x_max,x_sum,x_l1,xmma_min,xmma_max,xmma_sum,xmma_l1,' ...
    'df0dx_min,df0dx_max,df0dx_sum,df0dx_l1,dfdx_min,dfdx_max,dfdx_sum,dfdx_l1\n']);

loop = 0;
change = 1;
c_scale = NaN;
while change > 0.0001 && loop < maxiter
    loop = loop + 1;
    beta1 = min(beta1 + 0.1, 20);
    beta2 = min(beta2 + 0.1, 10);
    x(B) = bb;

    [eta1_x,eta2_x,eta1_y,eta2_y,eta1_z,eta2_z,rho1,xt1,xt2,yt1,yt2,zt1,zt2] = ...
        eta_ixyz(nelx_Pj,nely_Pj,nelz_Pj,x,nelx,nely,nelz,xf,yf,zf,beta1,LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var);
    eta = 0.5;
    rho = (tanh(beta2*eta)+tanh(beta2*(rho1-eta))) ./ ...
        (tanh(beta2*eta) + tanh(beta2*(1-eta)));
    drho_drho1 = beta2 * (1 - tanh(beta2 * (rho1 - eta)).^2) / ...
        (tanh(beta2 * eta) + tanh(beta2 * (1 - eta)));
    rho(loadele) = 1;
    drho_drho1(loadele) = 0;

    sK = reshape(KE(:)*(Emin+rho(:)'.^penal*(E0-Emin)),24*24*nelx*nely*nelz,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);

    ce = sum((U(edofMat)*KE).*U(edofMat),2);
    c = sum(sum((Emin+rho(:).^penal*(E0-Emin)).*ce));
    dc = -penal*(E0-Emin)*rho(:).^(penal-1).*ce;
    dv = ones(nely*nelx*nelz,1)/nelx/nely/nelz;
    dc = dc.*drho_drho1(:);
    dv = dv.*drho_drho1(:);
    [dx_beta1,dx_beta2,dx_deta1_y,dx_deta2_y,dx_deta1_z,dx_deta2_z] = ...
        dx_eta(eta1_x,eta2_x,eta1_y,eta2_y,eta1_z,eta2_z,nelx,nely,nelz,beta1,xf,yf,zf,xt1,xt2,yt1,yt2,zt1,zt2,LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var);

    df0dx = [dc'*dx_beta1,dc'*dx_beta2,dc'*dx_deta1_y,dc'*dx_deta2_y,dc'*dx_deta1_z,dc'*dx_deta2_z];
    dfdx = [dv'*dx_beta1,dv'*dx_beta2,dv'*dx_deta1_y,dv'*dx_deta2_y,dv'*dx_deta1_z,dv'*dx_deta2_z];
    df0dx = (H*df0dx')./Hs;
    dfdx = (H*dfdx')./Hs;
    v = sum(rho(:))/nelx/nely/nelz;
    fval = v - max(v*0.9,volfrac);
    f0val = c;
    if loop == 1
        c_scale = 1/f0val;
    end
    f0val = f0val*c_scale;
    df0dx = df0dx*c_scale;

    move = max(0.01 - 0.0001*loop, 0.001);
    mma.xmin = max(0,x-move);
    mma.xmax = min(1,x+move);
    dgt0 = 5;
    dgt = dgt0 - floor(log10([max(abs(df0dx(:))) max(abs(dfdx(:)))]));
    df0dx = round(df0dx*10^dgt(1))/10^dgt(1);
    dfdx = round(dfdx*10^dgt(2))/10^dgt(2);

    x_before = x;
    [x_mma,mma] = gensub(loop,f0val,df0dx,fval,dfdx,x,mma);
    x_change = norm(x_mma-x_before,'inf');
    sx = vec_stats(x_before);
    sxmma = vec_stats(x_mma);
    sdf0 = vec_stats(df0dx);
    sdf = vec_stats(dfdx);
    fprintf(fid, ['%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,' ...
        '%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,' ...
        '%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n'], ...
        loop,f0val,fval,v,beta1,beta2,move,c_scale,x_change, ...
        sx(1),sx(2),sx(3),sx(4),sxmma(1),sxmma(2),sxmma(3),sxmma(4), ...
        sdf0(1),sdf0(2),sdf0(3),sdf0(4),sdf(1),sdf(2),sdf(3),sdf(4));

    mma.xold2 = mma.xold1;
    mma.xold1 = x;
    x = x_mma;
    if loop > 5
        change = norm(x-mma.xold1,'inf');
    end
end
end

function s = vec_stats(v)
v = v(:);
s = [min(v), max(v), sum(v), sum(abs(v))];
end
