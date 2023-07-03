#pca skree function:

columnars = []

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pca_skree(df,columnars):
    """uses pandas df to do the skree plot on data using specified cols"""

    X = df[columnars]

    scaler = StandardScaler().fit(X)
    X_centered = scaler.transform(X)

    comps = len(columnars)

    ### PCA Scree plot
    n_comp = comps
    pca = PCA(n_components = n_comp, random_state = 1234)
    pca_data = pca.fit_transform(X_centered)

    fig, ax = plt.subplots(figsize = (15, 8))
    ax.plot([x for x in range(1, comps+1)], pca.explained_variance_ratio_, 'ro-', linewidth=2)
    ax.set_xticks([x for x in range(1, comps+1)])
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    return pca, scaler, X_centered
