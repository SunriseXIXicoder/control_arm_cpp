clear; close all; clc;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
matlab_dir = fullfile(repo_dir, '拔模SIMP对照组');
result_dir = fullfile(repo_dir, 'result');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
cd(matlab_dir);

summary_csv = fullfile(result_dir, 'trace_matlab_torsion_xy_mma_trace.csv');
vector_csv = fullfile(result_dir, 'trace_matlab_torsion_xy_mma_vectors.csv');
summary_fid = fopen(summary_csv, 'w');
vector_fid = fopen(vector_csv, 'w');
cleanup_obj = onCleanup(@() close_trace_files(summary_fid, vector_fid));
fprintf(summary_fid, ['iter,raw_compliance,f0val,fval,volume,beta1,beta2,move,c_scale,change,' ...
    'x_min,x_max,x_sum,x_l1,xmma_min,xmma_max,xmma_sum,xmma_l1,' ...
    'df0dx_min,df0dx_max,df0dx_sum,df0dx_l1,dfdx_min,dfdx_max,dfdx_sum,dfdx_l1\n']);
fprintf(vector_fid, 'iter,var,x,xmma,df0dx,dfdx\n');

DL=30;  DW=15; DH=15;
proport=1;
nelx=30*proport;  nely=15*proport;nelz=15*proport;
dx = [DL/nelx; DW/nely; DH/nelz];
[Eulerx0,Eulery0,Eulerz0] = meshgrid(dx(1)*(1:nelx),dx(2)*(1:nely),dx(3)*(1:nelz));
a=0.05;b=0.95;
xf=(Eulerx0-dx(1)/2)./DL;  xf=a+xf*(b-a);
yf=(Eulery0-dx(2)/2)./DW;  yf=a+yf*(b-a);
zf=(Eulerz0-dx(3)/2)./DH;  zf=a+zf*(b-a);
volfrac=0.2;
penal=1;
model='torsion';
maxiter=120;
E0 = 1;
Emin = 1e-9;
nu = 0.3;
[freedofs,iK,jK,KE,F,U,edofMat,loadele]=initiMesh(nelx,nely,nelz,model,nu);
opt=initiOpt();

nelx_Pj=nelx;
nely_Pj=nely;
nelz_Pj=nelz;
[H,Hs] = Sentivity_Filter(nelx_Pj,nely_Pj,nelz_Pj,max(min([nelx_Pj,nely_Pj,nelz_Pj]/3,5)));
[LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var]=SS(nelx,nely,nelz,DL,DW,DH,nelx_Pj,nely_Pj,nelz_Pj);
x1=zeros(nely_Pj*nelz_Pj,1);
x2=ones(nely_Pj*nelz_Pj,1);
y1=zeros(nelx_Pj*nelz_Pj,1);
y2=ones(nelx_Pj*nelz_Pj,1);
z1=zeros(nelx_Pj*nely_Pj,1);
z2=ones(nelx_Pj*nely_Pj,1);
x=[x1;x2;y1;y2;z1;z2];
beta1=10;
beta2=2;

loop = 0; change = 1;
mma = MMAInit(nelx_Pj,nely_Pj,nelz_Pj);
D=2*nelz_Pj*nely_Pj+2*nelz_Pj*nelx_Pj+1:2*nelz_Pj*nely_Pj+2*nelz_Pj*nelx_Pj+2*nelx_Pj*nely_Pj;
dd=x(D);

while change > 0.0001 && loop < maxiter
    loop = loop + 1;
    beta1=min(beta1+0.1,20);
    beta2=min(beta2+0.1,10);
    x(D)=dd;

    [eta1_x,eta2_x,eta1_y,eta2_y,eta1_z,eta2_z,rho1,xt1,xt2,yt1,yt2,zt1,zt2]=eta_ixyz(nelx_Pj,nely_Pj,nelz_Pj,x,nelx,nely,nelz,xf,yf,zf,beta1,LContro_E,FContro_E,DContro_E,L_Var,F_Var,D_Var);
    eta=0.5;
    rho = (tanh(beta2*eta)+tanh(beta2*(rho1-eta)))/(tanh(beta2*eta) + tanh(beta2*(1-eta)));
    drho_drho1 = beta2 * (0.1e1 - tanh(beta2 * (rho1 - eta)) .^ 2) / (tanh(beta2 * eta) + tanh(beta2 * (0.1e1 - eta)));
    rho(loadele)=1;
    drho_drho1(loadele)=0;

    sK = reshape(KE(:)*(Emin+rho(:)'.^penal*(E0-Emin)),24*24*nelx*nely*nelz,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
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
    fval=  sum(rho(:))/nelx/nely/nelz-max((sum(rho(:))/nelx/nely/nelz)*0.9,volfrac);
    v=sum(rho(:))/nelx/nely/nelz;
    f0val=c;
    if loop==1
        c_scale=1/f0val;
    end
    f0val = f0val*c_scale;
    df0dx =  df0dx*c_scale;

    move=max(0.01-0.0001*loop,0.001);
    mma.xmin=max(0,x-move);
    mma.xmax=min(1,x+move);

    dgt0 =5;
    dgt = dgt0 - floor(log10([max(abs(df0dx(:))) max(abs(dfdx(:)))]));
    df0dx = round(df0dx*10^dgt(1))/10^dgt(1);
    dfdx  = round(dfdx*10^dgt(2))/10^dgt(2);
    [x_mma,mma] = gensub(loop,f0val,df0dx,fval,dfdx,x,mma);
    mma.xold2=mma.xold1;
    mma.xold1=x;

    trace_change = change;
    if loop>5
        trace_change=norm(x_mma-mma.xold1,'inf');
    end
    fprintf(summary_fid, ['%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
        '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,' ...
        '%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n'], ...
        loop,c,f0val,fval,v,beta1,beta2,move,c_scale,trace_change, ...
        min(x),max(x),sum(x),sum(abs(x)),min(x_mma),max(x_mma),sum(x_mma),sum(abs(x_mma)), ...
        min(df0dx),max(df0dx),sum(df0dx),sum(abs(df0dx)),min(dfdx),max(dfdx),sum(dfdx),sum(abs(dfdx)));
    for qq = 1:numel(x)
        fprintf(vector_fid, '%d,%d,%.16e,%.16e,%.16e,%.16e\n', loop, qq, x(qq), x_mma(qq), df0dx(qq), dfdx(qq));
    end

    x=x_mma;
    change=trace_change;
    fprintf(' It.:%5i Obj.:%17.10e  Constriant.:%17.10e  V.:%17.10e ch.:%17.10e\n ',loop,f0val,fval,v,change);
end

function close_trace_files(summary_fid, vector_fid)
if summary_fid > 0
    fclose(summary_fid);
end
if vector_fid > 0
    fclose(vector_fid);
end
end
