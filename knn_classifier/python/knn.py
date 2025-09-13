# USAGE
# python knn.py --dataset dataset/animals

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
# if have many images, the 'jobs' help to tell scikit learn to use all the cores to compute the distances to all the points
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths  = list(paths.list_images(args["dataset"])) # obtains all the path of each image inside the three folders

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072)) # telling that each row will be a single image, and the 3072 means that will have a ...
# column for each rgb value. Because 32*32*3 = 3072

# show memory consumption
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024*1024.0)))

# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# divide into train/test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42)
# train and evaluate kNN classifier
print("[INFO] evaluating KNN")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY) # there is not learning in here, just safe data to memory
print(classification_report(testY, model.predict(testX), target_names=le.classes_))