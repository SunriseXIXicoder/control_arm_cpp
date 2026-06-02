function outFile = write_petsc_options(nx, ny, nz, outFile)
%WRITE_PETSC_OPTIONS Write a PETSc options file for the MPI solve kernel.
%
% Example:
%   write_petsc_options(700, 400, 120, '../petsc_100m.opts')
%
% Then run:
%   ./bin/control_arm_petsc -options_file petsc_100m.opts

if nargin < 1 || isempty(nx), nx = 700; end
if nargin < 2 || isempty(ny), ny = 400; end
if nargin < 3 || isempty(nz), nz = 120; end
if nargin < 4 || isempty(outFile), outFile = fullfile('..', 'petsc_case.opts'); end

fid = fopen(outFile, 'w');
if fid < 0
    error('write_petsc_options:openFailed', 'Cannot open %s for writing.', outFile);
end
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, '-nx %d\n', nx);
fprintf(fid, '-ny %d\n', ny);
fprintf(fid, '-nz %d\n', nz);
fprintf(fid, '-load 1.0\n');
fprintf(fid, '-ksp_type cg\n');
fprintf(fid, '-pc_type gamg\n');
fprintf(fid, '-pc_gamg_type agg\n');
fprintf(fid, '-mg_levels_ksp_type chebyshev\n');
fprintf(fid, '-mg_levels_pc_type jacobi\n');
fprintf(fid, '-ksp_rtol 1e-6\n');
fprintf(fid, '-ksp_max_it 500\n');
fprintf(fid, '-ksp_converged_reason\n');

dof = 3 * double(nx) * double(ny) * double(nz);
fprintf('Wrote %s for %.0f DOFs.\n', outFile, dof);
end
