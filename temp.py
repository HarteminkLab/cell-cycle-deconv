import numpy as np

def getWaveletKernel(type, N, par):
    return WavMat(MakeONFilter(type, par), N)

def WavMat(h, N, k0=None, shift=2):
    # WavMat -- Transformation Matrix of FWT_PO
    # Usage: W = WavMat(h, N, k0, shift)

    if k0 is None:
        k0 = int(np.log2(N))

    # Make QM filter G
    g = np.flip(h * (-1) ** np.arange(1, len(h) + 1))

    if not np.log2(N).is_integer():
        raise ValueError("N has to be a power of 2.")

    h = np.concatenate((h, np.zeros(N)))
    g = np.concatenate((g, np.zeros(N)))

    oldmat = np.eye(2**(int(np.log2(N)) - k0))

    for k in range(k0, 0, -1):
        ubJk = 2**(int(np.log2(N)) - k)
        ubJk1 = 2**(int(np.log2(N)) - k + 1)
        hmat = np.zeros((ubJk1, ubJk))
        gmat = np.zeros((ubJk1, ubJk))

        for jj in range(1, ubJk + 1):
            for ii in range(1, ubJk1 + 1):
                modulus = (N + ii - 2 * jj + shift) % ubJk1
                modulus = modulus + (modulus == 0) * ubJk1
                hmat[ii-1, jj-1] = h[modulus - 1]
                gmat[ii-1, jj-1] = g[modulus - 1]
        print("here")
        print(oldmat.shape)
        print(hmat.shape)
        print(gmat.shape)
        W = np.concatenate((np.matmul(oldmat, hmat), gmat.T), axis=1)
        oldmat = W

    return W

# Example usage:
# dat = np.array([1, 0, -3, 2, 1, 0, 1, 2])
# h = MakeONFilter('Haar', 99)
# W = WavMat(h, 2**3, 3, 2)
# wt = np.dot(W, dat)
# data = np.dot(W.T, wt)
# print(wt)
# print(data)

def MakeONFilter(Type, Par):
    # ... (previous code)

    if Type == 'Symmlet':
        if Par == 4:
            f = np.array([-0.107148901418, -0.041910965125, 0.703739068656,
                          1.136658243408, 0.421234534204, -0.140317624179,
                          -0.017824701442, 0.045570345896])
        elif Par == 5:
            f = np.array([0.038654795955, 0.041746864422, -0.055344186117,
                          0.281990696854, 1.023052966894, 0.896581648380,
                          0.023478923136, -0.247951362613, -0.029842499869,
                          0.027632152958])
        elif Par == 6:
            f = np.array([0.021784700327, 0.004936612372, -0.166863215412,
                          -0.068323121587, 0.694457972958, 1.113892783926,
                          0.477904371333, -0.102724969862, -0.029783751299,
                          0.063250562660, 0.002499922093, -0.011031867509])
        elif Par == 7:
            f = np.array([0.003792658534, -0.001481225915, -0.017870431651,
                          0.043155452582, 0.096014767936, -0.070078291222,
                          0.024665659489, 0.758162601964, 1.085782709814,
                          0.408183939725, -0.198056706807, -0.152463871896,
                          0.005671342686, 0.014521394762])
        elif Par == 8:
            f = np.array([0.002672793393, -0.000428394300, -0.021145686528,
                          0.005386388754, 0.069490465911, -0.038493521263,
                          -0.073462508761, 0.515398670374, 1.099106630537,
                          0.680745347190, -0.086653615406, -0.202648655286,
                          0.010758611751, 0.044823623042, -0.000766690896,
                          -0.004783458512])
        elif Par == 9:
            f = np.array([0.001512487309, -0.000669141509, -0.014515578553,
                          0.012528896242, 0.087791251554, -0.025786445930,
                          -0.270893783503, 0.049882830959, 0.873048407349,
                          1.015259790832, 0.337658923602, -0.077172161097,
                          0.000825140929, 0.042744433602, -0.016303351226,
                          -0.018769396836, 0.000876502539, 0.001981193736])
        elif Par == 10:
            f = np.array([0.001089170447, 0.000135245020, -0.012220642630,
                          -0.002072363923, 0.064950924579, 0.016418869426,
                          -0.225558972234, -0.100240215031, 0.667071338154,
                          1.088251530500, 0.542813011213, -0.050256540092,
                          -0.045240772218, 0.070703567550, 0.008152816799,
                          -0.028786231926, -0.001137535314, 0.006495728375,
                          0.000080661204, -0.000649589896])

        f = f / np.linalg.norm(f)

    return f

