# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from time import time
import matplotlib.pyplot as plt


CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []

    for path in tqdm(img_paths):
        # Load image
        img = Image.open(path)
        
        # Convert to grayscale first for better feature representation
        if img.mode != 'L':
            img = img.convert('L')
            
        # Apply center crop to maintain aspect ratio (optional)
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))
        
        # Resize to tiny image (16x16)
        img = img.resize((16, 16), Image.LANCZOS)  # LANCZOS resampling preserves more detail
        
        # Convert to numpy array
        img = np.array(img, dtype=np.float32)
        
        # Flatten the image
        img = img.flatten()
        
        # Zero-mean normalization improves robustness
        img = img - np.mean(img)
        
        # L2 normalization (unit vector)
        norm = np.linalg.norm(img)
        if norm > 0:
            img = img / norm
            
        # Add to feature list
        tiny_img_feats.append(img)

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 100
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    features = []
    for path in tqdm(img_paths):
        img = np.asarray(Image.open(path), dtype='float32')
        # Extract SIFT descriptors
        _, descriptor = dsift(img, step=[3,3], fast=True)
        # Randomly sample a fixed number of descriptors from each image
        # or sample a percentage of the descriptors
        if len(descriptor) > 0:
            # Option 1: Sample a fixed number (e.g., 50) from each image if available
            num_samples = min(400, len(descriptor))
            random_indices = np.random.choice(len(descriptor), num_samples, replace=False)
            sampled_descriptor = descriptor[random_indices]
            
            
            features.append(sampled_descriptor)

    # Concatenate all features
    features = np.concatenate(features, axis=0).astype('float32')

    # Further sampling from the combined feature set if it's still too large
    if len(features) > 5000000:  # Set an upper limit for the total number of descriptors
        random_indices = np.random.choice(len(features), 100000, replace=False)
        features = features[random_indices]

    print("Compute vocab...")
    start_time = time()
    # Cluster features into "vocab_size" groups
    vocab = kmeans(features, vocab_size, initialization="PLUSPLUS")

    end_time = time()
    print(f"It takes {((end_time - start_time)/60):.2f} minutes to compute vocab.")
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    image_feats = []

    for i, path in enumerate(tqdm(img_paths)):
        img = np.asarray(Image.open(path), dtype='float32')
        _, descriptor = dsift(img, step=[3,3], fast=True)
        
        if descriptor.shape[0] == 0:
            print(f"Warning: No SIFT features found for {path}")
            continue

        # 计算 SIFT 特征与视觉词汇的距离
        dist = cdist(descriptor, vocab, metric='correlation')

        idx = np.argmin(dist, axis=1)
        hist, _ = np.histogram(idx, bins=len(vocab))
        # normalize histogram
        hist_norm = [float(i)/sum(hist) for i in hist]

        image_feats.append(hist_norm)

        if i < 5:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(hist_norm)), hist_norm, color='blue', alpha=0.7)
            plt.xlabel("Visual Word Index")
            plt.ylabel("Frequency")
            plt.title(f"BoW Histogram for Image {i+1}")
            plt.savefig(f"D:/家愷的資料/大學/大三/電腦視覺/hw2_v2/p1/hist_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图表，释放内存

    return np.asarray(image_feats)

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list,
        k: int = 7
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []

    dist = cdist(test_img_feats, train_img_feats, metric='minkowski', p=0.2)

    for i in range(dist.shape[0]):
        # Get the indices of the k-nearest neighbors
        idx = np.argsort(dist[i])[:k]
        labels = [train_labels[j] for j in idx]
        weights = 1.0 / (dist[i][idx] + 1e-5)

        # 使用Counter计算加权投票
        from collections import Counter
        weighted_votes = Counter()
        for j, label in enumerate(labels):
            weighted_votes[label] += weights[j]

        # 选择权重最高的标签
        predicted_label = weighted_votes.most_common(1)[0][0]
        
        # Get the most frequent label among the k-nearest neighbors
        # predicted_label = max(set(labels), key=labels.count)
        #print(predicted_label)
        test_predicts.append(predicted_label)

    # Convert predicted class names to corresponding indices using CAT2ID
    # If pred_cats contains indices, you can map them back to strings
    #print(test_predicts)
    #test_predicts = [CAT[i] for i in test_predicts]

    return test_predicts
