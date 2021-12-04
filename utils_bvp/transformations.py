import numpy as np
import scipy.signal as sgn
import scipy.interpolate as interp


class RandomCrop(object):
    
    '''
    Takes numpy narray video with bounding box only and returns a randomly cropped selection video in numpy narray. 
    
    Parameters
    npy_bb_video : numpy narray with face only
    num_pixel_min : minimum number of pixels contained in the cropped box, to avoid noise if sample is too small   
    
    Returns
    patch : numpy narray with dims (num_frames, num_channels) after CHROM preprocessing
    '''
    
    def __init__(self, npy_bb_video):
        self.npy_bb_video = npy_bb_video
    
    @staticmethod
    def H_matrix(n, Lambda):
        '''
        Detrended using smoothness priors approach.
        '''
        I  = np.eye(n)
        D2 = np.zeros((n-2,n))
        for i in range(n-2):
            D2[i,i]   = 1
            D2[i,i+1] = -2
            D2[i,i+2] = 1
        return I - np.linalg.inv(I + Lambda**2*D2.T.dot(D2))
    
    @staticmethod
    def preproc_patch(patch, Lambda=10, fN=10):
        '''
        Implementation of CHROM preprocessing for video/patch selection. Works with
        both randomly_cropped_patch() and skin_cropped_patches().

        Parameters
        patch : numpy narray with dims (num_frames, y-crop, x-crop, num_channels) or
                list with same "dims" if crops do not have constant x or y dimension
        Lambda : H matrix parameter
        fN : Nyquist frequency for badpass filter

        Returns
        preprocessed_signal : numpy narray with dims (num_frames, num_channels)
        '''

        n = len(patch) # number of frames
        H = RandomCrop.H_matrix(n, Lambda)

        b, a = sgn.butter(2, (0.7/fN,3.5/fN),'bandpass')  #frequences normalisees

        if isinstance(patch, np.ndarray):
            zero_mean_signal = np.mean(patch, axis=(1,2)) - np.mean(patch, axis=(0,1,2))

        elif isinstance(patch, list):
            mean_per_frame = np.array([np.mean(p, axis=(0,1)) for p in patch])
            zero_mean_signal = mean_per_frame - np.mean(mean_per_frame, axis=0)

        preprocessed_signal = sgn.filtfilt(b, a, H.dot(zero_mean_signal), axis=0)

        return preprocessed_signal


    def __call__(self, num_pixel_min):
        
        video_size = self.npy_bb_video.shape[1] # bounding box always a square
        
        pos = np.random.randint(low=0, high=video_size, size=4)
        x1, x2 = sorted(pos[:2])
        y1, y2 = sorted(pos[2:])

        crop = self.npy_bb_video[0][y1:y2, x1:x2]

        while crop.shape[0]*crop.shape[1]<num_pixel_min:
            x1, x2, y1, y2 = sorted(np.random.randint(low=0, high=video_size, size=4))
            crop = self.npy_bb_video[0][y1:y2, x1:x2]

        patch = self.npy_bb_video[:, y1:y2, x1:x2, :]

        return RandomCrop.preproc_patch(patch)




