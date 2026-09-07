
import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

class hoSVD():
    """
        Tucker decomposition using High-Order SVD.
        input tensor was unfolded along each mode and decomposed with SVD. 
        The factor matrix U_k is truncated to the first n0 columns, where n0 is the minimal number of singule values (sigmas) 
             that makes $\\frac{\\sum_{i=1}^{n0} \\sigma_i}{\\sum_{i=1}^{n} \\sigma_i}$ > gamma, i.e. the cumsum of first n0 sigmas surpass 
             the given ratio (gamma) of the total sum of sigmas
        The core tensor is then calculated from the input X and each of the truncated factor matrix U_k

    """

    def __init__(self, gamma = 0.5):
        assert gamma > 0 and gamma < 1, "gamma has to be in the range of (0,1)."
        self.gamma = gamma
        self.data = None
        self.core = None
        self.factors = []
        self.n0s = []

    def _decompose(self, A):
        '''
        helper function to get the minimal number os sigule values (sigma) to satisfy:
            sum(sigma[:n0])/sum(sigma) > gamma
        
        no return value
        update self.n0s, self.factors
        '''
        u, s, _ = np.linalg.svd(A, full_matrices=False)
        tts = np.cumsum(s)
        n0 = np.where(tts/tts[-1] > self.gamma)[0].min() + 1
        U = u[:, :n0]
        self.n0s.append(n0)
        self.factors.append(U)

    def decompose(self, X):
        '''
        Decomposed core tensor and factor matrices
        update self.core, self.factors, self.n0s
        Return: None
        '''
        self.data = X
        self.factors = []
        self.n0s = []
        n = X.ndim
        G = X.copy()
        for i in range(n):
            Xn = tl.unfold(X, i)
            self._decompose(Xn)

            G = mode_dot(G, self.factors[-1].T, mode = i)
        self.core = G    

    def reconstruct(self):
        '''
        return reconstructed tensor
        ''' 
        assert self.core is not None, "Tensor has to be decomposed first."
        Xhat = self.core.copy()
        for i, m in enumerate(self.factors):
            Xhat = mode_dot(Xhat, m, mode = i)

        return Xhat

    @property
    def compress_ratio(self):
        '''
        get the compress ratio of the decomposed tensor
        '''
        assert self.core is not None, 'Tensor has to be decomposed first'
        compressed_size = self.core.size + np.sum([m.size for m in self.factors])
        compress_ratio = float(self.data.size)/compressed_size #((self.core.size + np.sum([m.size for m in self.factors])))
        return compress_ratio  

    @property        
    def error(self):
        '''
        return the reconstruction error
        '''
        Xhat = self.reconstruct()
        err = np.linalg.norm(self.data - Xhat)/np.linalg.norm(self.data)
        return err


if __name__ == "__main__":
    
    path = Path('./data', 'CaltechFaces')
    n_files = len(list(path.glob('*.jpg')))
    print(f'total images: {n_files}')

    imgs = []
    for f in path.glob('*.jpg'):
        image = np.array(Image.open(f))
        if image is not None:
            imgs.append(image)
    imgsX = tl.tensor(np.array(imgs))
    print(f'tensor shape: {imgsX.shape}')

   ## gamma vs compression ratio and re-construction error 
    errs = []
    rc_ratio = []
    gammas = np.linspace(0.1,1, 19)
    gammas[-1] = 0.99
    print(f'gammas to test:\n {gammas}')
    for i, gamm in enumerate(gammas):
        if not (i+1) % 5: print(f'{(i+1)*100/20}% processed.') 
        hs = hoSVD(gamma=gamm)
        hs.decompose(imgsX)
        err = hs.error
        cr = hs.compress_ratio
        errs.append(err)
        rc_ratio.append(cr)

        
    ## plot reconstruction error and compress ratio vs rank
    print('plot figures...')
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(gammas, errs, c = 'blue', label = 'Relative error')
    ax1.set_xlabel(f'gamma')
    ax1.set_ylabel('Relative error')
    col2 = 'tab:red'
    ax2 = ax1.twinx()
    ax2.plot(gammas, rc_ratio, c = col2, label = 'Compress ratio')
    ax2.tick_params(axis='y', labelcolor=col2)
    ax2.set_ylim((1, np.max(rc_ratio)))
    ax2.set_yscale('log')
    ax2.set_ylabel('Compress ratio', color = col2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Relative Error and Compress Ratio vs. gamma')
    plt.savefig('plots/hosvd_cr-and-err.vs.gamma.png')
    # plt.show()
    
    ## plot compress ratio vs relative error.
    plt.figure(figsize = (5,4))
    plt.plot(rc_ratio, errs)
    plt.scatter(rc_ratio, errs,s= 10, color = 'red', label = 'gramma')
    for i, gam in enumerate(gammas):
        va = 'top' if i == 0 else 'bottom'
        ha = 'left' if i == 0 else 'right'
        lab = 'gamma' if i == 0 else None
        plt.text(rc_ratio[i], errs[i], round(gam,2), color = 'red', fontsize = 8, ha = ha, va = va)
    plt.title('Compress ratio vs. Relative error')
    plt.xlabel('Compress ratio')
    plt.ylabel('Relative error')
    plt.xscale('log')
    plt.legend()
    plt.savefig('plots/hosvd_cr.vs.error.png')
    # plt.show()

    ## plot reconstructed images (gamma = 0.6)
    hs = hoSVD(gamma = 0.6)
    hs.decompose(imgsX)
    err = hs.error
    cr = hs.compress_ratio
    Xhat = hs.reconstruct()

    print(f'Overall Reconstruction error: {round(err, 2)}')
    print(f'Overall Compression ratio: {round(cr, 2)}')
    
    idx = np.random.choice(range(n_files), 10, replace = False)
    fig, ax = plt.subplots(2,10, figsize = (12, 2))
    for i, ii in enumerate(idx):
        err = np.linalg.norm(imgsX[ii, :, :] - Xhat[ii, :, :])/np.linalg.norm(imgsX[ii, :, :])
        err = round(err, 2)
        ax[0, i].imshow(imgsX[ii, :, :], cmap = 'gray')
        ax[0, i].axis('off')
        ax[0, i].text(25,100, f'err: {err}', color = 'yellow', fontsize=8)
        ax[0, i].set_title(f'Image {ii}')
        ax[1, i].imshow(Xhat[ii, :, :], cmap = 'gray')
        ax[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig('plots/hosvd-reconstructed-images.png')
    plt.show()
    