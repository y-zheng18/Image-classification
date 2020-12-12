import numpy as np
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

from matplotlib.colors import Normalize
import matplotlib.cm as cm

import matplotlib.gridspec as gridspec



def visualize(model, data, save_file):
    embedding_list = []
    label_list = []
    model.eval()
    use_gpu = torch.cuda.is_available()
    with torch.no_grad():
        for img, label in tqdm(data):
            if use_gpu:
                img = img.cuda()
            embeddings, _, _, _ = model(img)
            embedding_list.append(embeddings.cpu().numpy())
            label_list.append(label.numpy())
        embedding_list = np.concatenate(embedding_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
    print('fitting embeddings to 2 dim...')
    embeddings_2_dim = TSNE(n_components=2).fit_transform(embedding_list)
    print('fitting done!')
    print(embeddings_2_dim.shape)

    cmap = cm.Spectral
    norm = Normalize(vmin=0, vmax=np.max(label_list))
    colors = [cmap(norm(i)) for i in label]

    modelnet = {'dim0': embeddings_2_dim[:, 0], 'dim1': embeddings_2_dim[:, 1], 'y': label_list} # pd.DataFrame({'dim0': data[:, 0], 'dim1': data[:, 1], 'y': label})

    scatter = plt.scatter(
            x=modelnet["dim0"], y=modelnet["dim1"], c=colors,
            alpha=0.7, edgecolors='none'
            )

    plt.savefig('fig_tsne.pdf')
    plt.show()


