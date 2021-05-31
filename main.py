#!/usr/bin/python
import numpy as np
import os

from helpers import get_image_paths, load_images
from student import build_vocabulary, get_bags_of_words, svm_classify
from create_results_webpage import create_results_webpage


def classify():
    # This is the path the script will look at to load images from.
    data_path = './data/'

    # This is the list of categories / directories to use. The categories are
    # somewhat sorted by similarity so that the confusion matrix looks more
    # structured (indoor and then urban and then rural).
    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
                  'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
                  'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    # This list of shortened category names is used later for visualization.
    abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                       'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

    # Number of training examples per category to use. Max is 100. For
    # simplicity, we assume this is the number of test cases per category as
    # well.
    num_train_per_cat = 100

    # This function returns string arrays containing the file path for each train
    # and test image, as well as string arrays with the label of each train and
    # test image. By default all four of these arrays will be 1500x1 where each
    # entry is a string.
    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)

    print('Loading images.')
    train_images = load_images(train_image_paths)
    print("Training images loaded.")
    test_images = load_images(test_image_paths)
    print("Test images Loaded.")
    #   train_image_paths  1500x1   list
    #   test_image_paths   1500x1   list
    #   train_labels       1500x1   list
    #   test_labels        1500x1   list

    ############################################################################
    ## Step 1: Represent each image with the appropriate feature
    # The function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the
    # dimensionality of each image representation. See the starter code for
    # each function for more details.
    ############################################################################

    # Because building the vocabulary takes a long time, we save the generated
    # vocab to a file and re-load it each time to make testing faster. If
    # you need to re-generate the vocab (for example if you change its size
    # or the length of your feature vectors), simply delete the vocab.npy
    # file and re-run main.py
    if not os.path.isfile('vocab.npy'):
        print('No existing visual word vocabulary found. Computing one from training images.')

        # Larger values will work better (to a point), but are slower to compute
        vocab_size = 50

        # YOU CODE build_vocabulary (see student.py)
        vocab = build_vocabulary(train_images, vocab_size)
        np.save('vocab.npy', vocab)

    # YOU CODE get_bags_of_words.m (see student.py)
    print('Getting bag of words for training images')
    train_image_feats = get_bags_of_words(train_images)

    # You may want to write out train_image_features here as a *.npy and
    # load it up later if you want to just test your classifiers without
    # re-computing features
    print('Getting bag of words for test images')
    test_image_feats = get_bags_of_words(test_images)
    # Same goes here for test image features.

    ############################################################################
    ## Step 2: Classify each test image by training and using the svm classifier
    # The function to classify test features will return an N x 1 string array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels'. See the starter code for each function
    # for more details.
    ############################################################################

    # YOU CODE svm_classify (see student.py)
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
    ############################################################################

    ## Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section.

    # If we wanted to evaluate our recognition method properly we would train
    # and test on many random splits of the data. You are not required to do so
    # for this project.

    # This function will recreate results_webpage/index.html and various image
    # thumbnails each time it is called. View the webpage to help interpret
    # your classifier performance. Where is it making mistakes? Are the
    # confusions reasonable?
    ############################################################################

    create_results_webpage(train_image_paths,
                           test_image_paths,
                           train_labels,
                           test_labels,
                           categories,
                           abbr_categories,
                           predicted_categories)


if __name__ == '__main__':
    classify()
