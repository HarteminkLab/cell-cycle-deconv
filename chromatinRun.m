
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv')) 
addpath(genpath('analysis'))

outdir = 'output/genes';

% ------------------------------------------------------------------------------------

WT1_TP = [0	20	30	40	50	60	70	80	90	100	120	130	140];
WT2_TP = [0	20	30	40	50	60	70	80	90	100	120	130	140];

genename = 'CLB2';
gamma_val = 0.;
model = Model(genename, gamma_val);

clb2_chrom = h5read('/Users/trung/Research/chromatin-transformers/output/deconvolution/clb2_3x16.h5', '/data');
% permute the dimensions to match the original order in the NumPy array
clb2_chrom = permute(clb2_chrom, [4 3 2 1]);

deconvolvedClb2 = zeros(size(model.H, 2), size(clb2_chrom, 3), size(clb2_chrom, 4));

model.H = [];

% calculate H for WT1
model.timepoints = WT1_TP;
[H1, Hsegments, Hpos] = calcH(model);
model.Hsegments = Hsegments;
model.Hpos = Hpos;

% calculate H for WT2 and combine into a joint H
model.timepoints = WT2_TP;
[H2, Hsegments, Hpos] = calcH(model);
model.H = [H1' H2']';

model.timepoints1 = WT1_TP;
model.timepoints2 = WT2_TP;
model.timepoints = [WT1_TP WT2_TP];

model.gm = 0.02;

tic;

for row = 1:size(clb2_chrom, 3)
    for column = 1:size(clb2_chrom, 4)

        fprintf("%d %d, ", row, column);

        % Scale the occupancy data
        model.g1 = (clb2_chrom(:, 1, row, column)')*10;
        model.g2 = (clb2_chrom(:, 2, row, column)')*10;
        model.g = [model.g1 model.g2];
        
        try
            model = deconvolve(model);
        catch exception
            cvx_clear
            continue
        end

        deconvolvedClb2(:, row, column) = model.f;

    end
    fprintf("\n")
end

toc;

% drawDeconvolved(model);
filename = '/Users/trung/Research/chromatin-transformers/output/deconvolution/clb2_3x16_deconvolved.h5';
save_deconvolvedClb2 = permute(deconvolvedClb2, [3 2 1]);

% create an HDF5 file and dataset
h5create(filename, '/data', size(save_deconvolvedClb2), 'Datatype', class(save_deconvolvedClb2));

% write the MATLAB array to the HDF5 file
h5write(filename, '/data', save_deconvolvedClb2);

% --------------------

row = 2;
column = 5;

model.gm = 0.05;

% Scale the occupancy data so that the convex optimization works as it was
% calibrated for...
model.g1 = (clb2_chrom(:, 1, row, column)')*10;
model.g2 = (clb2_chrom(:, 2, row, column)')*10;
model.g = [model.g1 model.g2];

model = deconvolve(model);
drawDeconvolved(model);


