## Yelp Photo Classification Task

We need to predict the labels describing the business attributes of restaurants from the user submitted pictures. This is a Multi-Instance Multi-Label (MIML) classification problem. Each photo is assigned a business id and each business id is assigned multiple labels as given below.
The problem was launched as Kaggle competition. [10]

![Overview](/overview.png)

The original dataset contains nearly 2.3 lakh images taken by the users. Each photo is assigned a id called photoID. Each photoID is mapped to a businessID and each of the businessID is assigned multiple labels as shown in the figure above.

The different labels are listed below:
- 0: good_for_lunch
- 1: good_for_dinner
- 2: takes_reservations
- 3: outdoor_seating
- 4: restaurant_is_expensive
- 5: has_alcohol
- 6: has_table_service
- 7: ambience_is_classy
- 8: good_for_kids 



![Example](/example.png)


You can use the [editor on GitHub](https://github.com/akshaym96/Computer-Vision-Project/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

[//] ### Markdown

[//] Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

[//] ```markdown
[//] Syntax highlighted code block

[//] # Header 1
[//] ## Header 2
[//] ### Header 3

[//] - Bulleted
[//] - List

[//] 1. Numbered
[//] 2. List

[//] **Bold** and _Italic_ and `Code` text

[//] [Link](url) 
```




### Method-1 SIFT, Random Forest


In this method, features for both train and test images were extracted using SIFT. Then these descriptors were used to build clusters using K-Means. The clusters from K-Means are used to build the histograms for the train and test images. The extracted histograms are fed to 9 Random Forest Classifier for training and testing one 9 different available labels.


![RandomForest](/random_forest.png)


Feature Extraction time:-
- Train Images:-   15 hours 37 minutes 37 seconds
- Test Images:-     1 hour 22 minutes 26 secs.
- K-means clustering :-  3 hours  7 minutes 48 secs.
- Training time:-  1 minute 34 secs.
- Testing Time:-  30.5867 secs.
Results:- 
Accuracy:-
Strict Match:-  36.7031 %
One Mismatch:-  40.8372%
Two Mismatch:-  51.2016%

### Method-2  VGG + Fisher vectors

In this method, we use image features extracted by VGG model, run fisher vectors over the features to get the global descriptors and train a Multi-Output SVC which is used as the classifier. Since the data set is of varied image sizes, we are taking only 500*375*3 image sizes. There are 2 types of preprocessing on images were, one is direct resize to 224*224*3 and other method is taking 7 cropped images averages which overlap at various positions of the image.

### - VGG

The image features are extracted using a pre-trained VGG-16 model. In this model the last 3 layers have been removed mainly the fully connected layer, Relu and Dropout. So the outputs of the model are 4096. The architecture is as shown below.

![VGG](/VGG.png)


### - Fisher Vectors

Fisher Vectors was introduced in the paper, authored by F. Perronnin and C. Dance[1] and Florent Perronnin, Jorge Sánchez, and Thomas Mensink[2]. The FV is an image representation obtained by pooling local image features. It is frequently used as a global image descriptor in visual classification.
While the FV can be derived as a special, approximate, and improved case of the general Fisher Kernel framework, it is easy to describe directly. Let I=(x1,…,xN) be a set of D dimensional feature vectors (e.g. SIFT descriptors) extracted from an image. Let Θ=(μk,Σk,πk:k=1,…,K) be the parameters of a Gaussian Mixture Model fitting the distribution of descriptors. For each mode k, consider the mean and covariance deviation vectors. The FV of image I is the stacking of the vectors uk and then of the vectors vk for each of the K modes in the Gaussian mixtures.

![Fisher](/fisher.png)


For implementation , GGMM uses Cuda via CudaMat was used to generate the Gaussian Mixture Model[3][4].
Regarding Fisher Vector python implementation, there is no library implementation. So we used a reference implementation[5]  and modified it accordingly to fit the GGMM model.

![Fisher-VGG](/fisher-vgg.png)


### Method 3 - Res-Net + Multi-Output SVM Classifier

Res-Net50 and Res-Net10 was experimented and the behavior and performance of both were studied. Res-Net101 ran for 4.2 hours and Res-net50 took comparatively less time of 3.5 hours. In Res-Net101 model, a dropout layer of 0.2 and the linear layer was added in the end to prevent overfitting and to bring down the count of layers from 4096 to 2048 before feeding it to the model. The performance of Res-Net101 was higher than the Res-Net50. A slight increase in accuracy of 1.9% was seen. 

Features from the Res-Net was fed to the Multi-Output SVM model since they are less prone to overfitting. The Res-Net network is fine-tuned as above and can be fine-tuned some more in several different ways. It gives the user more freedom to access and modify the parts of the network which fairly affects the outcome. 


### Method 4 - SIFT + Ensemble model

This model uses Bag of visual words to extract the features using SIFT with a dense sampling of stride 50 and scale of 20. Then the image id is mapped with each of the business ids. Many pictures would have been taken in the same restaurant. So all these pictures will be mapped to the same business id and their labels. Then an ensemble model was built consisting of scikit-learn models (Decision tree Classifier, Random Forest Classifier, Nearest Neighbors, GradientBoostingClassifier, SGDClassifier). These sklearn models are fed to Multi-Output classifier since the labels in our problem is an array of values containing 0 and 1. Using an ensemble of models for predicting is proving to give better results than using a single model. The results of all the models are compared and the result predicted by most of the models is taken as the output. In case, all the models give a varied result, the prediction of the model with the highest accuracy is taken as the final prediction. But in these models, there is less power given to the user to work with the parameters which affect the outcome of each of the model. For this data and for the various Res-Net architectures and parameters I tried above, Res-Net performs better in this case.


### Method 5 - AlexNet + OneVsRestClassifier

In this method, we have used features from the penultimate linear layer which outputs 4096 dim vector.  These features are then fed to OneVsRestClassifier for training and testing. 

![alexNet](/alexnet.png)




For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### References

- [1] F. Perronnin and C. Dance. Fisher kernels on visual vocabularies for image categorization. In Proc. CVPR, 2006.
- [2] Florent Perronnin, Jorge Sánchez, and Thomas Mensink. Improving the fisher kernel for large-scale image classification. In Proc. ECCV, 2010.
- [3] Documentation of Ggmm - http://ebattenberg.github.io/ggmm/cpu.html
- [4] GMMs using CUDA (via CUDAMat) - https://github.com/ebattenberg/ggmm
- [5] Fisher Code is used as reference code and modified - https://github.com/jacobgil/pyfishervector
- [6] VLFeat - http://www.vlfeat.org/api/fisher.html
- [7] Scikit Learn - https://scikit-learn.org/
- [8] Pytorch tutorials
- [9] Numpy - https://docs.scipy.org/doc/numpy/reference/
- [10] Kaggle - https://www.kaggle.com/c/yelp-restaurant-photo-classification

### To-Dos
- [ ] Training a deep learning model from scratch.


Thanks for stopping by!! :D
