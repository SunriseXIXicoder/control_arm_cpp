clear; close all; clc;

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

loop = 1;
beta1=min(beta1+0.1,20);
beta2=min(beta2+0.1,10);
B=1:2*nelz*nely+2*nelz*nelx;
bb=x(1:2*nelz*nely+2*nelz*nelx);
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
c_scale=1/c;
df0dx=df0dx*c_scale;
dgt0 = 5;
dgt = dgt0 - floor(log10([max(abs(df0dx(:))) max(abs(dfdx(:)))]));
df0dx = round(df0dx*10^dgt(1))/10^dgt(1);
dfdx  = round(dfdx*10^dgt(2))/10^dgt(2);

out_path = fullfile(fileparts(mfilename('fullpath')), 'trace_matlab_mbb_once.csv');
fid = fopen(out_path, 'w');
fprintf(fid, 'compliance,volume,fval,c_scale,force_l1,force_nnz,freedofs,loadele_count,df0dx_min,df0dx_max,df0dx_sum,df0dx_l1,dfdx_min,dfdx_max,dfdx_sum,dfdx_l1\n');
fprintf(fid, '%.16e,%.16e,%.16e,%.16e,%.16e,%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n', ...
    c, v, fval, c_scale, full(sum(abs(F))), nnz(F), numel(freedofs), numel(loadele), ...
    min(df0dx), max(df0dx), sum(df0dx), sum(abs(df0dx)), min(dfdx), max(dfdx), sum(dfdx), sum(abs(dfdx)));
fclose(fid);
fprintf('wrote %s\n', out_path);
