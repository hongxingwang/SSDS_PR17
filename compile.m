function compile()
clear mex

mex -setup C

% Paths to Include Files and Library Files
lpath_to_matlab = '-LC:\Program Files\MATLAB\MATLAB Production Server\R2015a\extern\lib\win64\microsoft';
ipath_to_matlab = '-IC:\Program Files\MATLAB\MATLAB Production Server\R2015a\extern\include';
ipath_to_eigen = '-I.\SSDS\eigen3\';
paths = {lpath_to_matlab,ipath_to_matlab,ipath_to_eigen};

% Names of output folder and MEX-file
out_dir = './';
out_mex = 'fista_dykastra';
out_opts = {'-outdir', out_dir, '-output', out_mex};

% Enable OpenMP
com_opts = {'-O', '-largeArrayDims', 'COMPFLAGS="/openmp $COMPFLAGS"'};

% Source files to compile
files = {'./SSDS/mexFista.cpp', './SSDS/FISTA.cpp', './SSDS/proxflow.cpp', './SSDS/Solver.cpp', './SSDS/Utils.cpp', './SSDS/Graph.cpp'};

% mex
args = [com_opts,paths,out_opts,files];
mex(args{:})

