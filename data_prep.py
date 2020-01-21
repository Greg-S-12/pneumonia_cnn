import os, shutil
import pandas as pd
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from keras.utils import to_categorical


class Files():
#Create a class . object for which we can find and move files while keeping track of them

    def __init__(self,files=[]):
        self.files=files   # Create an empty list to store the files in our object.
        
        
    def find_files(self,file_directory,condition=None):
        
        self.file_directory = file_directory     # Keep track of the files
        self.files=[]    # Empty list so we manage this batch of files and can chain methods.
        
        # Will select all files in directory in no condition specified
        if condition == None:
            self.files += [file for file in os.listdir(self.file_directory)]       
        else:        
            self.files += [file for file in os.listdir(self.file_directory) if condition in file]   
        
        for file in self.files:
            if '.' in file:
                continue
            else:
                self.files.remove(file)
                print(file, 'Skipping file... File has no filetype: Invalid or directory.')
        
        return self
    
    
    def copy_files(self,destination_directory):
        """Will create copy of all files in current directory into a new one.
    
         dest_dir: string input, string of directory, all files will be copied to this direcory.
        """        
                        
        for file in self.files:
            origin = os.path.join(self.file_directory,  file)
            destination = os.path.join(destination_directory, file)
            shutil.move(origin,destination)
    
    def add_tag(self, tag):
        new_files=[]
        for file in self.files:
            file_string = file.split('.')
            file_name = file_string[0]
            file_type = str('.'+file_string[1])
            original = os.path.join(self.file_directory, file)
            new = os.path.join(self.file_directory, file_string[0]) + str('_{}'.format(tag))+file_type
            os.rename(original,new)
            new_files.append(file_name+str('_{}'.format(tag))+file_type)
        self.files=new_files
        return self
    
    def remove_tag(self, tag):
        new_files=[]
        for file in self.files:
            original = os.path.join(self.file_directory, file)
            new = original.replace(('_'+tag),'')
            os.rename(original,new)
            new_files.append(file.replace(('_'+tag),''))
        self.files=new_files
        return self
    
    
    def move_files(self, files, destination_directory):

        for file in files:
            origin = os.path.join(self.file_directory,  file)
            destination = os.path.join(destination_directory, file)
            shutil.move(origin,destination)
            
        self.file_directory = destination_directory    
        
    
    
    def train_test_val_split(self, splits, image_directory, subclasses=None):
            
        if len(splits)!=3:
            print('''Error: Must be equal to train, test and validation data splits respectively.''')
        if sum(splits)>1:
            print("Error: You are oversplitting your data -> Mixing of test/train/val data")
               
        n = (len(self.files))
        train_size = int(n*splits[0])
        test_size = int(n*splits[1])
        val_size = int(n*splits[2])
        
        self.subclasses = subclasses
        self.splits = splits
        self.train = self.files[:train_size]
        self.test = self.files[train_size:(n-val_size)]
        self.val = self.files[-val_size:]
        
        self.train_dir = '{}train/'.format(image_directory)
        self.test_dir = '{}test/'.format(image_directory)
        self.val_dir = '{}val/'.format(image_directory)
        
        folders=[self.train,self.test,self.val]
        directories=[self.train_dir,self.test_dir,self.val_dir]
        
        for directory in directories:
            if os.path.isdir(directory)==True:
                continue
            else:
                os.mkdir(directory)
        
        for folder, directory in list(zip(folders,directories)):
            self.move_files(folder,directory)  #self.file_directory becomes directory, ie self.train_dir
            for subfolder in subclasses:                
                if os.path.isdir('{}{}/'.format(directory,subfolder))==True:
                    continue
                else:
                    os.mkdir('{}{}/'.format(directory,subfolder))
            self.file_directory=image_directory
            
        for directory in directories:
            if len(subclasses)<=1:
                break            
            else:          
                for subclass in subclasses:
                    self.find_files(directory,subclass).move_files(self.files, os.path.join(directory, subclass))
                
        
        self.file_directory = None
        self.files = None


def find_files(file_directory,condition=None):
        
    file_directory = file_directory     # Keep track of the files
    files=[]    # Empty list so we manage this batch of files and can chain methods.
        
    # Will select all files in directory in no condition specified
    if condition == None:
        files += [file for file in os.listdir(file_directory)]       
    else:        
        files += [file for file in os.listdir(file_directory) if condition in file]   

        
    return files
        
    
def image_to_array(img_directory, image, label, number_of_classes):
    img = cv2.imread(os.path.join(img_directory,image))
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(label, number_of_classes)
    return img, label
        
def image_data_and_labels(imagetype, directory, subfolders, class_labels):
    """
    Reads in images from chosen directory, resizes, converts to RGB and rescales. Creates and returns two arrays: Image Data and corresponding class label.
    files: dictionary-like input
    imagetype: type of file to look for and convert e.g. '.jpeg', '.jpg', '.png', etc... If None, default to '.jpeg'
    directory: directory in which files are stored
    subfolders: Where files are stored

    """
    if imagetype==None:
        imagetype='.jpeg'
    if len(class_labels)!=len(subfolders):
        print('''Error: Class labels should correspond to subfolders. If files in different subfolders have the same label, this must be specified. 
        E.g. subfolders = ['healthy','viral_pneumonia','bacterial_pneumonia']
             class_labels = [0,1,1]''')
    data=[]
    labels=[]  
    for subfolder, class_label in list(zip(subfolders,class_labels)): 
        subfolder_directory = '{}{}'.format(directory,subfolder)    
        files = find_files(subfolder_directory, imagetype) # List of images             

        for image in files:
            img, label = image_to_array(subfolder_directory,image,class_label,len(set(class_labels)))
            data.append(img)
            labels.append(label)
    
    data=np.array(data)
    labels=np.array(labels).astype(int)
    
    return data,labels
    

    
def data_generator(data, batch_size, sequence, size, directory):
    
    n = len(data)
    steps = n//batch_size
    
    batch_images = np.zeros((batch_size, 224,224,3),dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2),dtype=np.float32)
    
    indices = np.arange(n)
    
    # Initialize a counter
    i = 0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['file']
            label = data.iloc[idx]['class']            
            
            # read the image and resize
            if "0001" in img_name:
                img = cv2.imread(str(os.path.join("{}normal".format(directory),str(img_name))))
            
            elif "virus" in img_name:
                img = cv2.imread(str(os.path.join("{}virus".format(directory),str(img_name))))
                                 
            else:
                img = cv2.imread(str(os.path.join("{}bacteria".format(directory),str(img_name))))
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            
            if img.shape[2]==1:       
                img = np.dstack([img, img, img])  # If grayscale then converts to rgb.
            
            # Setting the size of all the images as 224x224 - standard input size for VGG-16
            img = cv2.resize(img, size)
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            
            batch_images[count] = orig_img
            batch_labels[count] = encoded_label            

            # generating more samples of the undersampled class
            if (label==0) and (count < batch_size-3):
                aug_img1 = sequence.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = sequence.augment_image(img)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img2 = aug_img2.astype(np.float32)/255.
                
                batch_images[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_images[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                
                count +=3

            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_images, batch_labels
            
        if i>=steps:
            i=0    
            
