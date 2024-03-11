from skimage.feature import hog
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, pickle, glob, joblib, random


class HDR:
    def __init__(self):
        self.labels = []
        self.features = {}
    
    def load_image(self, path):
        # Read image as gray scale
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        # Resize image into 28*28
        resize_image = cv2.resize(image,(28,28))
        return resize_image

    def index_dataset(self, type):
        """Extract features and labels from images and serialize them."""
        if type == "train":
            features_path = "data/train_features.pkl"
            labels_path = "data/train_labels.pkl"
            dataset_folder = "dataset/train"
        else:
            features_path = "data/test_features.pkl"
            labels_path = "data/test_labels.pkl"
            dataset_folder = "dataset/test"

        if (not os.path.exists(features_path)) or (not os.path.exists(labels_path)):
            print("[INFO] Extraction features from images ...")
            for path, _ , files in os.walk(dataset_folder):
                for file in files:
                    try:
                        image_path = os.path.join(path, file)
                        image_label = file.split("_")[0]
                        image = self.load_image(image_path)
                        self.features[image_path] = self.extract_features(image) / 255.0
                        self.labels.append(int(image_label))
                    except Exception as e:
                        print(e)
                        pass
            self.save_data(features_path, labels_path)
        self.load_data(features_path, labels_path)

    def extract_features(self, img):
        """Compute Histogram of Oriented Gradients"""
        features = hog(img, orientations=4, pixels_per_cell=(4, 4), cells_per_block=(4, 4), block_norm = 'L2-Hys')
        return features

    def load_data(self, features_path, labels_path):
        """Load features and labels """
        print("[INFO] Loading Features and Labels ...")
        self.features = pickle.loads(open(features_path, "rb").read())
        self.labels = pickle.loads(open(labels_path, "rb").read())
    
    def save_data(self, features_path, labels_path):
        """Save features and labels """
        print("[INFO] Serializing Features and Labels ...")
        f = open(features_path, "wb")
        f.write(pickle.dumps(self.features))
        f.close()
        f = open(labels_path, "wb")
        f.write(pickle.dumps(self.labels))
        f.close()

    def train(self):
        self.index_dataset(type="train")
        # Split data into 80% train and 20% test subsets
        x_train, x_test, y_train, y_test = train_test_split(list(self.features.values()), self.labels, test_size = 0.20, random_state = 42)

        # checking the splits
        print('x_train shape: ', len(x_train))
        print('y_train shape: ', len(y_train))
        print('x_test shape: ', len(x_test))
        print('y_test shape: ', len(y_test))
        # Create a support vector classifier
        model = SVC(C=1.0, kernel='poly')

        # Evaluate accuracy scores by cross-validation.
        scores = cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1)
        print(scores)

        # Fit the SVM model according to the given training data.
        model.fit(x_train, y_train)

        # Predict the value of the digit on the test subset
        y_pred = model.predict(x_test)

        # Evaluate model
        print("Accuracy on test data: ", accuracy_score(y_test, y_pred), "\n")

        # Showing the main classification metrics (Precision, recall and F-measure)
        scores = classification_report(y_test, y_pred, labels=model.classes_)
        print(scores)

        # Compute confusion matrix to evaluate the accuracy of a classification.
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        print(cm)

        # save model
        joblib.dump(model, 'model/svm.model', compress=3)

    def test(self):
        self.index_dataset(type="test")
        model = joblib.load('model/svm.model')
        predictions = model.predict(list(self.features.values()))
        # Evaluate model
        print("Accuracy on test data: ", accuracy_score(self.labels, predictions), "\n")

        temp = list(zip(self.features.keys(), predictions, self.labels))
        random.shuffle(temp)
        images, predictions, labels = zip(*temp)

        true_predictions = {}
        false_prediction = {}
        # Get 5 false and true prediction labels
        
        for image, prediction, label in zip(images,predictions, labels):
            if prediction != label:
                false_prediction[image] = prediction
            else:
                true_predictions[image] = prediction
            if len(true_predictions) == 5 and len(false_prediction) == 5:
                break
        fig1, axes1 = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
        fig2, axes2 = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))

        fig1.suptitle('True Predictions', fontsize=16)
        for ax, image, prediction in zip(axes1, true_predictions.keys(), true_predictions.values()):
            ax.set_axis_off()
            image = plt.imread(image)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prediction: {prediction}")

        fig2.suptitle('False Predictions', fontsize=16)
        for ax, image, prediction in zip(axes2, false_prediction.keys(), false_prediction.values()):
            ax.set_axis_off()
            image = plt.imread(image)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prediction: {prediction}")
        
        plt.show()

if __name__ == "__main__":
    hdr = HDR()
    hdr.train()
    # hdr.test()