alpha = 26;
model_diff = deconvolve('DSE3', 'diff_g1', 'wt1', [alpha], 0, 0, 0, 0);
model_normal = deconvolve('DSE3', 'normal', 'wt1', [alpha], 0, 0, 0, 0);

sn_diff = square_pos(norm(model_diff.H*model_diff.f./model_diff.g-1, 2));
sn_normal = square_pos(norm(model_normal.H*model_normal.f./model_diff.g-1, 2));

sn_diff
sn_normal

nf = model_normal.f;
d = nf(1:128);
c = nf(129:256);
r = nf(257:end);

% post G1 .. leave as it is
% CG1 .. set to zero

G1 = model_diff.lengths(7)*model_diff.lengths(2);
CG1_len = G1+alpha;
RG1_len = model_diff.lengths(1)+alpha;
postG1_len = model_diff.lengths(2)-G1-alpha;
DG1_len = model_diff.lengths(3)+G1;

postG1_p = ceil(128*G1/(postG1_len+G1));

CG1 = rescale([1:postG1_p], c([1:postG1_p]), (postG1_p-1)/127);
POSTG1 = rescale([postG1_p+1:128], c([postG1_p+1:128]), (128-postG1_p)/128);

RG1_p = ceil(model_diff.lengths(1)/RG1_len*128);
RG1_1 = rescale([1:128], r, 128/RG1_p);
RG1_2 = rescale([1:postG1_p], c([1:postG1_p]), postG1_p/(128-RG1_p-1));
RG1 = [RG1_1 RG1_2];

DG1_p = ceil(model_diff.lengths(3)/DG1_len*128);
DG1_1 = rescale([1:128], d, 128/DG1_p);
DG1_2 = rescale([1:postG1_p], c([1:postG1_p]), postG1_p/(128-DG1_p-1));
DG1 = [DG1_1 DG1_2];

newf = [POSTG1 DG1 CG1 RG1];

new_sn_diff = square_pos(norm(model_diff.H*newf'./model_diff.g-1, 2));
new_sn_diff
