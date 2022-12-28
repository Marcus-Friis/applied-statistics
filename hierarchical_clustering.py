import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.rcsetup import cycler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy import stats


df = pd.read_csv('data/kandidater_data.csv').dropna()

agg = df.groupby('CurrentPartyCode').agg(
    {str(i): ['mean'] for i in range(1, 26)}
).reset_index()


# clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=24)
clusterer = AgglomerativeClustering(n_clusters=14, compute_distances=True)
clusters = clusterer.fit_predict(df[[str(i) for i in range(1, 26)]])
df['cluster'] = clusters

# Apply t-SNE to the selected columns
tsne = TSNE(n_components=2, random_state=69420)
X_tsne = tsne.fit_transform(df[[str(i) for i in range(1, 26)]])

# Add the transformed coordinates to the existing dataframe
df['tsne1'] = X_tsne[:, 0]
df['tsne2'] = X_tsne[:, 1]

colormap = {'A': '#A82721', 
            'O': '#EAC73E', 
            'V': '#254264', 
            'Ø': '#E6801A', 
            'I': '#3FB2BE', 
            'Å': '#2B8738', 
            'B': '#522170',   # '#733280', 
            'F': '#E07EA8', 
            'C': '#96B226', 
            'D': '#127B7F', 
            'K': '#8B8474', 
            'Q': '#C49414', 
            'M': '#943CA4',  # '#832B93' 
            'Æ': '#2C5877'}
df['PartyColor'] = df.CurrentPartyCode.map(colormap)

plt.style.use('ggplot')

fig = plt.figure(figsize=(15,7))
gs = GridSpec(2, 2, figure=fig)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(df.tsne1, df.tsne2, color=df.PartyColor, label=df.CurrentPartyCode, alpha=.6)
ax.set_title('True parties dim-reduced with t-sne')
ax.set_xlabel('t-sne 1')
ax.set_ylabel('t-sne 2')


ax = fig.add_subplot(gs[1, 0])
# color_palette = sns.color_palette('Paired', 14)
color_palette = {
    0: 'V',
    1: 'Ø',
    2: 'K',
    3: 'M',
    4: 'Q',  # Nyt parti, blevet til Q
    5: 'B',
    6: 'D',
    7: 'O',
    8: 'C',
    9: 'Å',
    10: 'Æ',
    11: 'F',
    12: 'I',
    13: 'A'
}
color = [colormap[color_palette[i]] for i in df.cluster]

ax.scatter(df.tsne1, df.tsne2, color=color, alpha=.6)
ax.set_title('Clusters dim-reduced with t-sne')
ax.set_xlabel('t-sne 1')
ax.set_ylabel('t-sne 2')

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

def llf(id):
    if id < df.shape[0]:
        return df.CurrentPartyCode.iloc[id]
    else:
        return str(id)

mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#988ED5',  '#348ABD', '#E24A33', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8'])

ax = fig.add_subplot(gs[:, 1])
params = {'ax': ax, 'leaf_label_func': llf, 'leaf_rotation': 0, 'show_leaf_counts': True}
plot_dendrogram(clusterer, truncate_mode="level", **params)
ax.grid(False)
ax.set_yticks([])
ax.set_title('Hierarchical clustering of party means')
ax.set_xlabel('Candidates')

xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(colormap[lbl.get_text()])

ax.set_xticks([])

plt.tight_layout()
plt.savefig('tsne_dendogram.svg')
