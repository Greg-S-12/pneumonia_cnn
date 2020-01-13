import os, shutil


def make_classes(splits, list_of_classes):
    
    """Will create folders and subfolders in current directory according to input '(splits, list_of_classes):'
    Based on how data is to be split and number of prediction classes for storing data. Intended for data in image recognition and for training of CNN.
    
     splits: [] , list-like, string input, corresponding to file names user wants parent files to be named. Will create folders in which class folders are stored.
     list_of_classes:  [] , list-like input of predictor classes. Will create subfolders for each item in split.

Example
-------
    For binary classification with train, test and validation datasets:
     make_classes(['train','test','val'], ['class_1','class_2']
     Creates folders: 'train','test','val', each with subfolders: 'class_1','class_2'
    
    """   
    for split in splits:
        os.mkdir(f'{split}/')
        for c in list_of_classes:
            sub_folder = os.path.join(f'{split}/', f'{c}')
            os.mkdir(sub_folder)

            
def copy_files(orig_dir, dest_dir):
    
    """Will create copy all files in a directory to a new one.
    
     orig_dir: string input, string of directory, all files will be copied from this directory.
     dest_dir: string input, string of directory, all files will be copied to this direcory.
    """        
    files = [file for file in os.listdir(orig_dir)]
    
    for file in files:
        origin = os.path.join(orig_dir,file)
        destination = os.path.join(dest_dir,file)
        shutil.copyfile(origin,destination)
        
        
def find_files(directory, condition):
    
    files = [file for file in os.listdir(directory) if condition in file]    
    return files

def move_files(directory, files, destination):
        
    for file in files:
        origin = os.path.join(directory,  file)
        destination_file = os.path.join(destination, file)
        shutil.move(origin,destination_file)

        
def train_test_val(split_percentages, files):
    
    train_size = int(len(files)*split_percentages[0])
    test_size = int(len(files)*split_percentages[1])
    val_size = int(len(files)*split_percentages[2])
    
    train = []
    test = []
    val = []
    
    for file in files[:train_size]:
        train.append(file)
    for file in files[train_size:(len(files)-val_size)]:
        test.append(file)
    for file in files[-val_size:]:
        val.append(file)
    
    return train, test, val
                
def move_files_to_groups_by_split(split_percentages, files, groups, directory, destination_folder):
    
    for group in groups:
        splits={}
        splits['train'], splits['test'], splits['val'] = train_test_val(split_percentages, files[group])
        for split in splits:
            destination = f'{destination_folder}/{split}/{group}/'
            move_files(directory, splits[split], destination)
        
        
    

        
    
    