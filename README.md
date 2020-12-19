# Forged Signature Detection using Convolutional Neural Networks.

This project aims at performing signature verification and forgery detection, which is the process of verifying signatures automatically to determine whether a signature is genuine or not. It performs signature verification by using convolutional neural networks (CNNs). The model is based on AlexNet architecture. The signature images are pre-processed in a batch by batch manner and are split into training and testing data. The images have been re-sized and randomly cropped to 227 x 227 which is the default input size for AlexNet. 

### Dataset

In this project, a combination of the SigComp2009 dataset from the ICDAR 2009 signature verification competition and the CEDAR Signature database were used. The SigComp 2009 dataset contains NISDCC dataset which is expected to have 1920 images from 12 authentic writers (5 authentic signatures per writer) and 31 forging writers (5 forgeries per authentic signature) but from the public dataset, only 1898 images are available and the dataset collected at the Netherlands Forensic Institute (NFI) which consists of authentic signatures from 100 newly introduced writers (each writer wrote his signature 12 times) and forged signatures from 33 writers (6 forgeries per signature). 

CEDAR signature database contains signatures of 55 signers belonging to various cultural and professional backgrounds. Each of these signers signed 24 genuine signatures 20 minutes apart. Each of the forgers tried to emulate the signatures of 3 persons, 8 times each, to produce 24 forged signatures for each of the genuine signers. Hence the dataset comprises 55 × 24 = 1320 genuine signatures as well as 1320 forged signatures.

### Instructions to run:
There are 4 python files, 
1. Def_CNN.py - AlexNet layers are defined.
2. train_network.py - Code to train the model
3. test_network.py - Code to test the saved model
4. evaluation.py - To calculate evaluation metrics.

#### Run the Code:
**Training:**
1. Run the train_network.py file.
2. Please enter the path to the downloaded dataset on the system.
3. Please enter the path of where the model needs to be saved.
4. The model will be trained and saved in the specified location. 'Loss plot.png', 'accuracy_values.csv', 'loss_values.csv' files are created in the same folder as the project.

**Testing:**
5. Run the test_network.py file.
6. Please enter the system path of the saved model (.pkl file).
7. Please enter the path to the downloaded dataset on the system.
8. 'Confusion_matrix.png' and 'accuracy_test.csv' files are created and the evaluation metrics will be printed on the console.



