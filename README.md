# Diagnosing Pneumonia from X-ray Images with Convolutional Neural Networks
This project involves Convolutional Neural Networks as a method of image recognition in order to identify cases of pneumonia from x-ray images.
<br> Neural Networks have recently been employed as a method of identifying cancer from x-ray screenings or CAT scans [1,2]. Companies are now leveraging deep learning and machine learning techniques for healthcare, such as BenevolentAI's 'Benevolent Platform', in order to: 
<br><br>• Improve healthcare services
<br>• Increase diagnosis accuracy and speed
<br>• Identify new/best treatments 
<br>• Improve treatment efficacy

Pneumonia is the #1 infection-related cause of death in developed countries and is the cause of death for 15% of children aged under 5 years in devoloping countries. Viral and bacterial Pneumonia have different treatments and the cost of misdiganosis is often high, leading to increased mortality rates. Radiology is the most succesful method of diagnosis but is time consuming and costly.
<br> This project aims to diagnose penumonia from x-rays with a high rate of recall, reducing the cost, resources and time in the procedure, resulting in reduced mortality rates.

## Methods
Convolutional Neural Networks have been employed using Keras with Tensorflow backend. Models were built using various layer types, activation functions and transfer learning in order to experiment with different architecture. This consisted of:

### Layers
•Convolutional2D
<br>•MaxPooling2D
<br>•SeparableConv2D (Pointwise and Depthwise Convolutions)
<br>•BatchNormalization
<br>•Dropout (varying dropout percentages)

### Activation Functions
•SoftMax (for final output layer)
<br>•ReLu (Rectified Linear Unit)
<br>•Swish (modified sigmoid - see: https://arxiv.org/pdf/1710.05941v1.pdf)

### Transfer Learning
• VGG16 architecture, varying number of first layers implemented. These were then 'frozen', so that the weights for these layers are not trainable.
<br>• VGG16 was used as it has been trained on 15 million images on imagenet.
<br>• The first few layers are able to identify different 'blob' sizes, colours and edges.

(In order to perform the above easily, one should convert all images which are in grayscale to RGB such that all images need to be of the same colour scale.)

## Different Libraries
•Python (programming language projected written in)
<br>•Keras (Neural Network)
<br>•ImgAug (data generation and image augmentation)
<br>•cv2/OpenCV (Read images, convert colourscale, resize and normalize)
<br>•Pandas (Dataframes and managing labels)
<br>•NumPy (Arrays for data and labels, calculations)
<br>•Seaborn (Visualizations)
<br>•os/shutil (Managing files and moving them to correct directories/subfolders)

The Imgaug library has several requirements which will be installed alongside Imgaug. These libraries can all be safely installed with Pip or Conda (if using the Conda environment).

## Project Description

### The Data
The dataset conists of 5,863 x-ray images of children ages 1-5 from Guangzhou Children's Medical Center, and can be found here: https://data.mendeley.com/datasets/rscbjbr9sj/2 .The images were graded and labeled by two experts in order to diagnose Pneumonia and the type before a final check by a third expert. There were 3 labels - normal (healthy lungs), bacterial pneumonia and viral pneumonia. Viral and bacterial pneumonia are the two most common types. The cause can also be fungal but this is very rare and not present within this dataset. Other causes of pneumonia can be serious conditions such as heart failure, or a pulmonary embolism, which must be indentified rapidly for successful treatment.
<br> As these files are quite large, it is recommended you either add them to your gitignore or use git LFS (Large File Storage). However as models for the proejct were run on Google Cloud git LFS may not function correctly. The data can downloaded from here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Hypothesis and Goal
The first goal is to see if we can successfully separate cases of pneuomiona from healthy lungs and make correct predictions from just the images. Radiology is the most effective method of diagnosis and implementing an AI in place of doctors would save time and resources and could potentially achieve a higher accuracy than a single expert.
<br> Secondly it was of interest to see if we could further separate cases of pneumonia from viral or bacterial infection as the treatment (and thereby survival rate) are dependant on the type of infection. Bacterial pneumonia can be treated with anti-biotics and the maximum time from admission to begin treatment is 4 hours before mortality rates begin to increase. Viral pneumonia is usually only identified through failure of treatment using anti-biotics, which, in very young or elderely patients, puts them at higher risk of death. It can also lead to the original cause  of symptoms (which can be a severe condition) untreated. There are several methods, aside from radiology, which can be used alongside traditional diagnosis methods (chest exminations, listening to breathing) such as:

<br>• Blood cultures - these can take 24-48 hours to cultivate, so not suitable for high risk patients
<br>• Measuring highly-sensitive C-reactive protein (hs-CRP) - not a consitently accurate predictor [3]
<br>• Measuring Procalcitonin levels (PCT) - succssful at separating bacterial pneumonia vs differential diagnoses, but not specifically for identifying viral infections.[3]

If a high rate of succesful predictions is achieved, this can be implemented, increasing the use of radiology by signifcantly reducing the cost and time of this method and decreasing mortality rates overall.

### Convolutional Neural Networks
Convolutional Neural Networks rely on the same methodology as all neural networks - back-propagation. This is the process of the network calculating various 'weights' for each neuron in each layer and outputting a prediction. Based on whether the predictions given are correct or not, their associated errors are calculated and the model 'back-propagates' this error, adjusting the weights for each layer in order to improve the model. This utilises Gradient Descent to adjust the weights.
<br> CNNs provide a great method for image recognition. The images can be read in as a tensor of RGB pixels (a matrix for each colour) on a scale of 0-255.  This is then normalized by a factor of 255 such that our neural network treats them equally (a larger range in pixel intensity garners a larger weighting) before resizing the image to a smaller resolution such that processing is less intensive (faster). By breaking down the image into smaller and smaller pieces via 2D stepwise convolution and pooling, the network is trained to identify different aspects of an image. This will initally be blobs and edges but can then begin to identify whole images based from this training. 

### Transfer Learning with VGG16
 Transfer learning is a great technique for image recognition. There are pre-trained models, such as MobileNet or VGG16, which have been trained on millions of images, for over a thousand classes, over several weeks and can identify a large number of images correctly. By implementing the first few layers and freezing them (so that the weights are not trainable and don't change) a model can be built off of these layers, with the aim of reducing training requirements (number of images, time) and increasing accuracy.
<br> In order to transfer learn from VGG16, the input of the images is required to be 224x224x3 (resolution x number of channels). For this project, images were resized to accordingly and converted to RGB via the cv2 library.

### Class Imbalance and Data Generation
In this dataset, as in many medical datasets, there is a class imbalance. This means that classes which the netowrk is trying to predict have a substantial difference in terms of the amount of training data available for each class. In this case, there are far more cases of Pneumonia that healthy lungs (as one might expect considering the patients have undergone an x-ray scan to determine the cause of an illness/sypmtom). What this means is our trined model will learn to expect more of one class than the other and can begin to predict accordingly. This will also invalidate accuracy as an evaluation metric. 
<br> The simplest method to tackle this problem is weighting the classes, such that the undersampled class is more heavily weighted, reducing the bias in training our model.
<br> Another method which can be used in image recognition is data generation. Here the images from the undersampled class are taken and augmented (rotated, cropped, flipped, contras/brightness can be increased/decreased etc.) and these augmented images are used in place of the original. This process is repeated until there are near equal number of each class in the training dataset. In this project, the data generator is used during training - this is known as 'on-the-fly' data generation, as the data is not actually stored locally.








## References
[1] McKinney, S.M., Sieniek, M., Godbole, V. et al. International evaluation of an AI system for breast cancer screening. Nature 577, 89–94 (2020) doi:10.1038/s41586-019-1799-6
<br>[2] B. Parmadean et al. Transfer Learning from Chest X-Ray Pre-trained Convolutional Neural Network for Learning Mammogram Data, (Procedia Computer Science Volume 135, 2018, Pages 400-407) https://doi.org/10.1016/j.procs.2018.08.190
<br>[3] Müller, B., Harbarth, S., Stolz, D. et al. Diagnostic and prognostic accuracy of clinical and laboratory parameters in community-acquired pneumonia. BMC Infect Dis 7, 10 (2007) doi:10.1186/1471-2334-7-10