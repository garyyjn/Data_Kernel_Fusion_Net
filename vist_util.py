from matplotlib import pyplot as plt
import numpy as np
def showone(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def showbunch(data, width_limit = 4):
    batch = data.shape[0]
    fig, axes = plt.subplots(int(batch/width_limit) + 1, width_limit)
    for i in range(batch):
        axes[int(i/width_limit)][i%width_limit].imshow(data[i], interpolation='nearest')
        axes[int(i/width_limit)][i%width_limit].get_xaxis().set_visible(False)
        axes[int(i/width_limit)][i%width_limit].get_yaxis().set_visible(False)
    plt.show()