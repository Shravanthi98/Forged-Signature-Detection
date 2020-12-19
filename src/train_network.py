# import all the necessary packages
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from numpy import savetxt
import torch.nn.functional as F
import os
import numpy as np
from Def_CNN import *
import matplotlib.pyplot as plt


# Define parameters used in the Dataloader class
# Batch size = number of images that are trained together as a batch
BATCH_SIZE = 32
# number of iterations
EPOCHS = 40
# Transformations performed on every image in the train set
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(227),
    transforms.RandomCrop(227),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
    ])


# Main function
if __name__ == '__main__':
    # Path where the dataset is stored on the system
    DATA_PATH = input("Please enter the path for the dataset")
    # Path to save the model
    SAVE_PATH1 = input("Please enter the path for saving the model")
    # Instantiation of CNN
    alexnet1 = Net()
    # Loading the dataset
    # Referred from https://pytorch.org/docs/stable/torchvision/datasets.html
    total_data = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=TRANSFORM_IMG)
    # true labels
    targets = total_data.targets
    # Splitting the data into train and test set in the ratio of 8:2
    # Referred from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(total_data, targets, test_size=0.2, shuffle=True, random_state=101)
    # Loading the shuffled train data
    train_data_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # create optimizer
    # Referred from https://pytorch.org/docs/stable/index.html
    optimizer = torch.optim.Adam(params=alexnet1.parameters(), lr=0.0005, weight_decay=0.0005)
    print('Optimizer created')
    print("Training AlexNet")
    total_steps = 1
    total = 0
    correct = 0
    # To accumulate accuracy and loss values from all the iterations
    loss_values = []
    accuracy_values = []
    # Training starts
    for epoch in range(EPOCHS):
        running_loss = 0
        running_accuracy = 0
        for imgs, lbs in train_data_loader:
            output = alexnet1(imgs)
            # Cross-entropy loss function for the predicted output
            loss = F.cross_entropy(output, lbs)
            running_loss += loss.item() * imgs.size(0)

            # update the parameters
            loss.backward()
            optimizer.step()
            # Checkpoint for every 100 steps
            if total_steps % 100 == 0:
                for name, parameter in alexnet1.named_parameters():
                    if parameter.grad is not None:
                        grad = torch.mean(parameter.grad)
                        print('\t{} - grad: {}'.format(name, grad))
                    if parameter.data is not None:
                        weight = torch.mean(parameter.data)
                        print('\t{} - param: {}'.format(name, weight))
            optimizer.zero_grad()
            # Printing the loss for every 10 steps
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    total += lbs.size(0)
                    accuracy = torch.sum(preds == lbs)
                    correct += accuracy.item()

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                          .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
            total_steps += 1
        print('Accuracy of the model on the train images: %f %%' % (100 * correct / total))
        running_accuracy = (100 * correct / total)
        total = 0
        correct = 0
        loss_values.append(running_loss / len(X_train))
        accuracy_values.append(running_accuracy)
    # Path to save the model
    final_path = os.path.join(SAVE_PATH1, 'alexnet_bs32_0.0005.pkl')
    # Saving the model
    torch.save(alexnet1, final_path)
    loss_array = np.array(loss_values)
    accuracy_array = np.array(accuracy_values)
    # Logging the loss and accuracy values into text files
    savetxt('loss_values.csv', loss_array, delimiter=';')
    savetxt('accuracy_values.csv', accuracy_array, delimiter=';')
    fig = plt.figure()
    plt.plot(loss_values)
    plt.show()
    fig.savefig('Loss_plot.png')

    # End of the program
    exit(0)





