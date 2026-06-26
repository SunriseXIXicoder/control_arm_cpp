clear; close all; clc;

steps_env = getenv('MBB_TRACE_STEPS');
if isempty(steps_env)
    max_steps = 80;
else
    max_steps = str2double(steps_env);
    if isnan(max_steps) || max_steps < 1
        error('MBB_TRACE_STEPS must be a positive integer.');
    end
    max_steps = floor(max_steps);
end
write_vectors = ~strcmp(getenv('MBB_TRACE_WRITE_VECTORS'), '0');
write_final_vtk = strcmp(getenv('MBB_TRACE_WRITE_FINAL_VTK'), '1');

desktop_dir = fullfile(getenv('USERPROFILE'), 'Desktop');
dirs = dir(fullfile(desktop_dir, '*SIMP'));
is_target = false(numel(dirs), 1);
for q = 1:numel(dirs)
    is_target(q) = dirs(q).isdir && ~startsWith(dirs(q).name, 'PIML');
end
target_dirs = dirs(is_target);
if isempty(target_dirs)
    error('Could not find the MATLAB SIMP directory on the Desktop.');
end
matlab_dir = fullfile(desktop_dir, target_dirs(1).name);
addpath(matlab_dir);
cd(matlab_dir);

DL=10; DW=10; DH=10;
proport=2;
nelx=10*proport; nely=10*proport; nelz=10*proport;
dx = [DL/nelx; DW/nely; DH/nelz];
[Eulerx0,Eulery0,Eulerz0] = meshgrid(dx(1)*(1:nelx),dx(2)*(1:nely),dx(3)*(1:nelz));
a=0.05; b=0.95;
xf=(Eulerx0-dx(1)/2)./DL; xf=a+xf*(b-a);
yf=(Eulery0-dx(2)/2)./DW; yf=a+yf*(b-a);
zf=(Eulerz0-dx(3)/2)./DH; zf=a+zf*(b-a);
volfrac=0.2;
penal=1;
model='mbb';
E0 = 1;
Emin = 1e-9;
nu = 0.3;

[freedofs,iK,jK,KE,F,U,edofMat,loadele]=initiMesh(nelx,nely,nelz,model,nu);
[H,Hs] = Sentivity_Filter(nelx,nely,nelz,max(min([nelx,nely,nelz]/3,5)));
[LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var]=SS(nelx,nely,nelz,DL,DW,DH,nelx,nely,nelz);

x1=zeros(nely*nelz,1);
x2=ones(nely*nelz,1);
y1=zeros(nelx*nelz,1);
y2=ones(nelx*nelz,1);
z1=zeros(nelx*nely,1);
z2=ones(nelx*nely,1);
x=[x1;x2;y1;y2;z1;z2];
beta1=10;
beta2=2;
loop = 0;
change = 1;
mma = MMAInit(nelx,nely,nelz);
B=1:2*nelz*nely+2*nelz*nelx;
bb=x(1:2*nelz*nely+2*nelz*nelx);

tool_dir = fileparts(mfilename('fullpath'));
summary_path = fullfile(tool_dir, 'trace_matlab_mbb_steps_summary.csv');
vector_path = fullfile(tool_dir, 'trace_matlab_mbb_steps_vectors.csv');
fs = fopen(summary_path, 'w');
if write_vectors
    fv = fopen(vector_path, 'w');
else
    fv = -1;
end
fprintf(fs, ['iter,raw_compliance,f0val,volume,fval,c_scale,beta1,beta2,move,change,' ...
             'x_min,x_max,x_sum,x_l1,x_weighted,x_sqsum,' ...
             'xmma_min,xmma_max,xmma_sum,xmma_l1,xmma_weighted,xmma_sqsum,' ...
             'df0dx_min,df0dx_max,df0dx_sum,df0dx_l1,df0dx_weighted,df0dx_sqsum,' ...
             'dfdx_min,dfdx_max,dfdx_sum,dfdx_l1,dfdx_weighted,dfdx_sqsum\n']);
if write_vectors
    fprintf(fv, 'iter,var,x,xmma,df0dx,dfdx\n');
end

while change > 0.0001 && loop < max_steps
    loop = loop + 1;
    beta1=min(beta1+0.1,20);
    beta2=min(beta2+0.1,10);
    x(B)=bb;

    [eta1_x,eta2_x,eta1_y,eta2_y,eta1_z,eta2_z,rho1,xt1,xt2,yt1,yt2,zt1,zt2]=eta_ixyz(nelx,nely,nelz,x,nelx,nely,nelz,xf,yf,zf,beta1,LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var);
    eta=0.5;
    rho = (tanh(beta2*eta)+tanh(beta2*(rho1-eta)))/(tanh(beta2*eta) + tanh(beta2*(1-eta)));
    drho_drho1 = beta2 * (1 - tanh(beta2 * (rho1 - eta)) .^ 2) / (tanh(beta2 * eta) + tanh(beta2 * (1 - eta)));
    rho(loadele)=1;
    drho_drho1(loadele)=0;

    sK = reshape(KE(:)*(Emin+rho(:)'.^penal*(E0-Emin)),24*24*nelx*nely*nelz,1);
    K = sparse(iK,jK,sK);
    K = (K+K')/2;
    U(:) = 0;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    ce = sum((U(edofMat)*KE).*U(edofMat),2);
    c = sum(sum((Emin+rho(:).^penal*(E0-Emin)).*ce));
    dc = -penal*(E0-Emin)*rho(:).^(penal-1).*ce;
    dv = ones(nely*nelx*nelz,1)/nelx/nely/nelz;
    dc = dc.*drho_drho1(:);
    dv = dv.*drho_drho1(:);
    [dx_beta1,dx_beta2,dx_deta1_y,dx_deta2_y,dx_deta1_z,dx_deta2_z]= dx_eta(eta1_x,eta2_x,eta1_y,eta2_y,eta1_z,eta2_z,nelx,nely,nelz,beta1,xf,yf,zf,xt1,xt2,yt1,yt2,zt1,zt2,LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var);

    df0dx=[dc'*dx_beta1,dc'*dx_beta2,dc'*dx_deta1_y,dc'*dx_deta2_y,dc'*dx_deta1_z,dc'*dx_deta2_z];
    dfdx=[dv'*dx_beta1,dv'*dx_beta2,dv'*dx_deta1_y,dv'*dx_deta2_y,dv'*dx_deta1_z,dv'*dx_deta2_z];
    df0dx=(H*df0dx')./Hs;
    dfdx=(H*dfdx')./Hs;
    v=sum(rho(:))/nelx/nely/nelz;
    fval=v-max(v*0.9,volfrac);
    f0val=c;
    if loop==1
        c_scale=1/f0val;
    end
    f0val=f0val*c_scale;
    df0dx=df0dx*c_scale;

    moveNow=max(0.01-0.0001*loop,0.001);
    mma.xmin=max(0,x-moveNow);
    mma.xmax=min(1,x+moveNow);

    dgt0=5;
    dgt=dgt0-floor(log10([max(abs(df0dx(:))) max(abs(dfdx(:)))]));
    df0dx=round(df0dx*10^dgt(1))/10^dgt(1);
    dfdx=round(dfdx*10^dgt(2))/10^dgt(2);
    [x_mma,mma]=gensub(loop,f0val,df0dx,fval,dfdx,x,mma);
    mma.xold2=mma.xold1;
    mma.xold1=x;
    if loop>5
        change=norm(x_mma-mma.xold1,'inf');
    end

    ids=(1:numel(x))';
    stats = @(a) [min(a), max(a), sum(a), sum(abs(a)), sum(a(:).*ids), sum(a(:).*a(:))];
    sx=stats(x);
    sxm=stats(x_mma);
    sd0=stats(df0dx);
    sdv=stats(dfdx);
    fprintf(fs, ['%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
                 '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
                 '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
                 '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
                 '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n'], ...
        loop,c,f0val,v,fval,c_scale,beta1,beta2,moveNow,change, ...
        sx(1),sx(2),sx(3),sx(4),sx(5),sx(6), ...
        sxm(1),sxm(2),sxm(3),sxm(4),sxm(5),sxm(6), ...
        sd0(1),sd0(2),sd0(3),sd0(4),sd0(5),sd0(6), ...
        sdv(1),sdv(2),sdv(3),sdv(4),sdv(5),sdv(6));

    if write_vectors
        for q = 1:numel(x)
            fprintf(fv, '%d,%d,%.16e,%.16e,%.16e,%.16e\n', loop, q, x(q), x_mma(q), df0dx(q), dfdx(q));
        end
    end
    fprintf('trace mbb iter %d/%d f0=%g v=%g change=%g\n', loop, max_steps, f0val, v, change);
    x=x_mma;
end

fclose(fs);
if write_vectors
    fclose(fv);
end
if write_final_vtk
    [X, Y, Z] = meshgrid(1:nelx, 1:nely, 1:nelz);
    vtk_path = fullfile(tool_dir, sprintf('trace_matlab_mbb_loop_%d.vtk', loop));
    vtkwrite(vtk_path, 'structured_grid', X, Y, Z, 'scalars', 'density', rho);
    fprintf('wrote %s\n', vtk_path);
end
fprintf('wrote %s\n', summary_path);
if write_vectors
    fprintf('wrote %s\n', vector_path);
end
