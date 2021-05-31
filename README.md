# bag-of-visual-words_image-classification
A python implementation of multi-class image classification using bag of visual words technique and SVM classifier.

Description:
Training images are used to build a bag of words by extracting features from each image using Histogram of Oriented Gradients(hog) algorithm.
The features extracted from all the training images are then clustered, using Kmean, into the desired number of clusters to create the bag of words.

The dataset is created by using the obtained bag of words to create a histogram of features for each image in the training images. The histogram 
is then normalized and the label of each image is obtained, using the directory name of the image.

SVM classifier is trained using the dataset that we created, and then tested using test images. The process of creating test dataset is the same as the
process described for the training data.

A confusion matrix is produced using the test results and the accuracy, mean of confusion matrix diagonal, is calculated.

Instructions:
Use [link](https://drive.google.com/file/d/1hbKp679nDRoc3vR5sGTBns70Le7d6c_8/view?usp=sharing) to download the dataset. Unzip the file in the
same directory as the scripts.

Run main.py.
