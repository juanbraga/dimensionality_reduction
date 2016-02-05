# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 13:06:48 2016

@author: juan.braga
"""

#%% loading and computing features UrbanSound dataset 
from features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import scipy.stats
import csv
import os.path

def compute_features(filename):

    fs,audio_array = wav.read(filename)
    
    mfcc_25 = mfcc(audio_array,samplerate=fs,winlen=0.064,winstep=0.032,numcep=25,
               nfilt=40,nfft=512,lowfreq=0,highfreq=fs/2,preemph=0,
               ceplifter=0,appendEnergy=True)
    
    first = np.diff(mfcc_25, axis=0)
    second = np.diff(first, axis=0)
    
    minimum = np.amin(mfcc_25,axis=0)
    maximum = np.amax(mfcc_25,axis=0)
    median = np.median(mfcc_25,axis=0)
    mean = np.mean(mfcc_25,axis=0)
    variance = np.var(mfcc_25,axis=0)
    skewness = scipy.stats.skew(mfcc_25,axis=0)
    kurtosis = scipy.stats.kurtosis(mfcc_25,axis=0)
    first_mean = np.mean(first,axis=0)
    first_variance = variance = np.var(first,axis=0)
    second_mean = np.mean(second,axis=0)
    second_variance = variance = np.var(second,axis=0)
     
    features = np.concatenate((minimum, maximum, median, mean, variance, skewness, kurtosis, first_mean, first_variance, second_mean, second_variance), axis=0)
    
    return features
    
caract = ['minimo', 'maximo', 'mediana', 'media', 'varianza', 'skewness', 'kurtosis', 'media_1Derivada', 'varianza_1Derivada', 'media_2Derivada', 'varianza_2Derivada']
clases = ['gun_shot', 'siren', 'car_horn', 'children_playing', 'dog_bark', 'air_conditioner', 'jackhammer', 'drilling', 'engine_idling', 'street_music']

cr = csv.reader(open("//CTRIAS/audio_dataset/UrbanSound8K/metadata/UrbanSound8K.csv","rb"))
next(cr) 
      
X=np.empty([275,0])
y=[]
i=0

for row in cr:
    
    i = i + 1
    filename = os.path.join('//CTRIAS/audio_dataset/audio_dataset_16bits/', 'fold' + row[5] + '/' + row[0])
    print filename
    features = compute_features(filename)
    X_aux = np.c_[X,features]
    X = X_aux    
    y.append(row[7])

n_features, n_sample = X.shape
n_neighbors = 30

np.save('muestras', X)
np.save('etiquetas',y)
np.save('clases', clases)
np.save('caract', caract)                
                
#%%

print(__doc__)
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_in, y, title=None):
    x_min, x_max = np.min(X_in, 0), np.max(X_in, 0)
    X_in = (X_in - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X_in.shape[0]):
        if (y[i]==clases[0]):
            color = 0
        if (y[i]==clases[1]):
            color = 1
        if (y[i]==clases[2]):
            color = 2
        if (y[i]==clases[3]):
            color = 3
        if (y[i]==clases[4]):
            color = 4
        if (y[i]==clases[5]):
            color = 5
        if (y[i]==clases[6]):
            color = 6
        if (y[i]==clases[7]):
            color = 7
        if (y[i]==clases[8]):
            color = 8
        if (y[i]==clases[9]):
            color = 9
        plt.text(X_in[i, 0], X_in[i, 1], '*',
                 color=plt.cm.Set1(color / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

#    if hasattr(offsetbox, 'AnnotationBbox'):
#        # only print thumbnails with matplotlib > 1.0
#        shown_images = np.array([[1., 1.]])  # just something big
#        for i in range(X.shape[0]):
#            dist = np.sum((X_in[i] - shown_images) ** 2, 1)
#            if np.min(dist) < 4e-3:
#                # don't show points that are too close
#                continue
#            shown_images = np.r_[shown_images, [X_in[i]]]
#            imagebox = offsetbox.AnnotationBbox(
#                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                X_in[i])
#            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#%%

#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")



#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))

#%%

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components
X = np.load('muestras.npy')
y = np.load('etiquetas.npy')
clases = np.load('clases.npy')
caract = np.load('caract.npy')
print("Computing Linear Discriminant Analysis projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2.transpose(), y)
plot_embedding(X_lda, y,
               "Linear Discriminant projection of the digits (time %.2fs)" %
               (time() - t0))

#%%

#----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding(X_iso,
               "Isomap projection of the digits (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle,
               "Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle,
               "Modified Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle,
               "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa,
               "Local Tangent Space Alignment of the digits (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds,
               "MDS embedding of the digits (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding(X_reduced,
               "Random forest embedding of the digits (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X)

plot_embedding(X_se,
               "Spectral embedding of the digits (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()