import numpy as np



def gloria(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
   
    series = (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[...,np.newaxis].astype(np.float32)




print(gloria(4,1))