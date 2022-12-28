import matplotlib as mpl
from matplotlib.rcsetup import cycler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


df = pd.read_csv('data/kandidater_data.csv').dropna()

agg = df.groupby('CurrentPartyCode').agg(
    {str(i): ['mean'] for i in range(1, 26)}
).reset_index()


# clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=24)
clusterer = AgglomerativeClustering(n_clusters=14, compute_distances=True)
clusters = clusterer.fit_predict(agg[[str(i) for i in range(1, 26)]])
agg['cluster'] = clusters

colormap = {'A': '#A82721', 
            'O': '#EAC73E', 
            'V': '#254264', 
            'Ø': '#E6801A', 
            'I': '#3FB2BE', 
            'Å': '#2B8738', 
            'B': '#733280', 
            'F': '#E07EA8', 
            'C': '#96B226', 
            'D': '#127B7F', 
            'K': '#8B8474', 
            'Q': '#C49414', 
            'M': '#832B93', 
            'Æ': '#2C5877'}
df['PartyColor'] = df.CurrentPartyCode.map(colormap)

plt.style.use('ggplot')

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# def llf(id):
#     if id < 14:
#         return agg.CurrentPartyCode.iloc[id]
#     else:
        # return str(id)

def llf(id):
    if id < df.shape[0]:
        return agg.CurrentPartyCode.iloc[id]
    else:
        return str(id)



mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#988ED5',  '#348ABD', '#E24A33', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8'])

fig, ax = plt.subplots()
agg = df.groupby('CurrentPartyCode').agg({f'{i}': 'mean' for i in range(1, 26)}).reset_index()

# clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=24)
# clusters = clusterer.fit_predict(agg[[str(i) for i in range(1, 26)]])

params = {'ax': ax, 'leaf_label_func': llf, 'leaf_rotation': 0, 'show_leaf_counts': True}
plot_dendrogram(clusterer, truncate_mode="level", **params)
ax.grid(False)
ax.set_yticks([])
ax.set_title('Hierarchical clustering of party means')
ax.set_xlabel('Parties')

xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(colormap[lbl.get_text()])

ax.invert_xaxis()

plt.savefig('dendogram_parties.svg')
