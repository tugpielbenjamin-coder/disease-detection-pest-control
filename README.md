Potato Leaf Disease Detection

A convolutional neural network (CNN) built with TensorFlow/Keras to classify potato leaf images into three categories: Early Blight, Late Blight, and Healthy. The goal is to support early, low-cost identification of common potato leaf diseases from photographs, which can help farmers and agricultural extension workers act before an outbreak spreads.

Overview

Potato crops are highly susceptible to fungal diseases such as early blight and late blight, both of which can cause significant yield loss if not caught early. This project trains an image classification model that takes a photo of a potato leaf and predicts whether it is healthy or affected by one of the two diseases, along with a confidence score.

Dataset
Classes (3): Potato___Early_blight, Potato___Late_blight, Potato___healthy
Input size: 256 × 256 RGB images
Split: 80% train / 10% validation / 10% test, split by batch after shuffling
Batch size: 32
Approach
Data pipeline – Images are loaded with tf.keras.preprocessing.image_dataset_from_directory, then cached, shuffled, and prefetched for performance.
Preprocessing & augmentation – Images are resized/rescaled to [0, 1], with random horizontal/vertical flips and random 90°-multiple rotations applied during training to improve generalization.
Model architecture – A CNN built with the Keras Functional API: a stack of Conv2D + MaxPooling2D blocks followed by dense layers, with the resize/rescale and augmentation steps embedded directly in the model graph via Lambda layers.
Training – Compiled with the Adam optimizer and sparse categorical cross-entropy loss, trained for 10 epochs on the training/validation split.
Evaluation – Assessed on the held-out test set using accuracy/loss, a full classification report (precision, recall, F1), and a confusion matrix heatmap.
Model export – The trained model is versioned and saved as a .keras file, alongside a JSON file mapping class indices to class names, for later inference.
Repository Structure
├── notebooks/
│   └── potatoes_Leaves_disease_detection.ipynb   # Main training & evaluation notebook
├── models/                                        # Saved model versions (.keras) + class mappings (generated)
└── README.md
Requirements
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
Seaborn
scikit-learn

Install dependencies with:

bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
Usage
Place the potato leaf image dataset in a directory named potatost/, organized into one subfolder per class.
Open notebooks/potatoes_Leaves_disease_detection.ipynb and run the cells in order to:
Load and visualize the dataset
Train the CNN model
Evaluate performance on the test set (accuracy, classification report, confusion matrix)
Save the trained model and class labels to models/
To run inference with a saved model, load it with tf.keras.models.load_model("models/<version>.keras") along with its corresponding <version>_classes.json class mapping.
Results

The model was trained for 10 epochs and evaluated on a held-out test set, with performance tracked via accuracy/loss curves, a classification report, and a confusion matrix. See the notebook's training and evaluation sections for the full metrics and visualizations, including sample predictions with confidence scores.

Note: Current results indicate the model would benefit from further tuning (e.g., more epochs, learning rate adjustment, additional data, or transfer learning) to improve test accuracy — noted here as a direction for future work.

Future Improvements
Increase training epochs and tune hyperparameters
Experiment with transfer learning (e.g., MobileNet, ResNet, EfficientNet)
Expand the dataset with more diverse images and additional disease classes
Deploy the model behind a simple API or web app for real-time predictions
