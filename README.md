# Dog Breed Classification
The goal of this project is to build a pipeline to process real-world, user-supplied images in the case of dog breed images classification. Given an image of a dog or a human, the algorithm will identify an estimate of its canine’s breed.

This project is based on the [Machine Learning Engineer Udacity Course](https://eu.udacity.com/course/machine-learning-engineer-nanodegree--nd009).

## Project Highlights
This project is designed to give a hands-on experience with Convolutional Neural Network and how to build a pipeline to process real-world and user-supplied images.

Things learned by completing the project:

	How to build a pipeline to process real-world and user-supplied images
	How to engineer a real-world application that involves solving many problems without a perfect answer
	How to create an imperfect solution will nonetheless creating a fun user experience

This project contains several files:

    dog_breed.ipynb: This is the main file where the project is performed.
    dogImages : This folder contain dog images with which the CNN will train and test on.
    lfw : This folder contain human images.
    haarcascades : This folder contain OpenCV's implementation of [Haar feature-based cascade classifiers to detect human faces](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) in images.
    extract_bottleneck_features.py : This file is used to obtain the bottleneck features corresponding to the chosen CNN architecture.
    
## Software and Libraries
This project uses the following software and Python libraries:

- [Python](https://www.python.org/download/releases/3.0/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)(v0.17)
- [matplotlib](http://matplotlib.org/)

The softwares will need to be installed and ready to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If Python is not installed yet, it is highly recommended to install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/anielwong/dog_breed_classifier.git
cd dog_breed_classifier
```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location `path/to/dog_breed_classifier/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location `path/to/dog_breed_classifier/lfw`. If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Create two folders named 'bottleneck_features' and 'saved_models'.

5. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog_breed_classifier/bottleneck_features`.

6. Download the [Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog_breed_classifier/bottleneck_features`.

7. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

8. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog_breed_classifier
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog_breed_classifier
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog_breed_classifier
	```

9. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog_breed_classifier python=3.5
	source activate dog_breed_classifier
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog_breed_classifier python=3.5
	activate dog_breed_classifier
	pip install -r requirements/requirements.txt
	```
	
10. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
11. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog_breed_classifier environment. 
```
python -m ipykernel install --user --name dog_breed_classifier --display-name "dog_breed_classifier"
```

13. Open the notebook.
```
jupyter notebook dog_breed_classifier.ipynb
```

14. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog_breed_classifier environment by using the drop-down menu (**Kernel > Change kernel > dog_breed_classifier**). Then, open the notebook.



