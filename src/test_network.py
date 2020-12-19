# import all the necessary packages
from train_network import *
from evaluation import *
from numpy import savetxt
from sklearn import metrics
import itertools

# Batch size = number of images that are tested together as a batch
BATCH_SIZE = 1
# Path to load the saved model
SAVE_PATH2 = input("Please enter the path of the saved model")
SAVE_PATH = os.path.join(SAVE_PATH2, "alexnet_bs32_0.0005.pkl")

# Transformations performed on every image in the test set
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
    ])

# main function
if __name__ == '__main__':
    # Path where the dataset is stored on the system
    DATA_PATH = input("Please enter the path for the dataset")
    # Loading the dataset
    # Referred from https://pytorch.org/docs/stable/torchvision/datasets.html
    total_data = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=TRANSFORM_IMG)
    targets = total_data.targets
    # Referred from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(total_data, targets, test_size=0.2, shuffle=True, random_state=101)
    # Loading the test data
    test_data_loader = DataLoader(X_test, batch_size=1, shuffle=False, num_workers=0)
    # y_true : A list containing the expected values of a prediction
    # y_pred : A list containing the predicted values from the network.
    y_true = []
    y_pred = []
    # Load the saved model from the specified path
    alexnet1 = torch.load(SAVE_PATH)
    # Setting the model in evaluation mode
    alexnet1.eval()
    total = 0
    correct = 0
    total_steps = 1
    accuracy_list = []
    for imgs, lbs in test_data_loader:
        output = alexnet1(imgs)
        _, preds = torch.max(output, 1)
        y_pred.append(preds)
        y_true.append(lbs)
        total = total + 1
        accuracy = torch.sum(preds == lbs)
        accuracy_list.append(accuracy)
        correct += accuracy.item()
    print('Accuracy of the model on the test images: %f %%' % (100 * correct / total))
    data = np.array(accuracy_list)
    # Logging the accuracy values into a text file
    savetxt('accuracy_test.csv', data, delimiter=';')
    # Generating a confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Function call to calculate the evaluation metrics from the confusion matrix
    confusion_metrics(cm)
    # Saving the confusion matrix plot to the system
    # Referred from https://stackoverflow.com
    thresh = cm.max() / 2
    fig = plt.figure()
    cmap = plt.get_cmap('Blues')
    plt.matshow(cm, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix.png')
# End of the program
exit(0)
