y = [1.5 3];
c = [1 2 2; 2 4 4];

ACE2 = [669.120304 526.1310512	451.2681587	414.7134934	417.8857089	585.1562459	1201.177103	2183.027163	2016.687169	1242.273246	836.8962575	951.4543493	1251.218156	1448.752109	1259.408901	1285.295097];

%WAVETYPE = 'Haar';
%WAVEPAR = 2;

WAVETYPE = 'Daubechies';
WAVEPAR = 12;

%WAVETYPE = 'Symmlet';
%WAVEPAR = 5;

g = ACE2';
len = numel(g);
W = getWaveletKernel(WAVETYPE, len, WAVEPAR);

gamma = 1;

cvx_begin

	variable c(len);

	minimize(...
		square_pos(norm(g-W*c,2))+gamma*norm(W*c,1)...
	);

	subject to
		c>=0;
cvx_end

%figure;
%plot(c);
figure;
plot(ACE2,'b');
hold on;
plot(W*c,'r');
