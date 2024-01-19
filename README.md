# TensorFlow CNN for Chess Piece Detection

This is an investigation into creating a CNN (Convolutional Neural Network) to classify 3D pictures of chess pieces into one of 13 categories (2 of the following for white and black creates 12 classes, and one extra class for empty square).

- Pawn
- Rook
- Knight
- Bishop
- Queen
- King

I created a CNN model in TensorFlow, and leveraged transfer learning on the VGG16 model trained on ImageNet to improve performance and prevent overfitting. I implemented 2 convolutional layers and one Global max pooling layer to flatten the neurons before the classification head, which consisted of a dense layer with the softmax activation.

I implemented many different algorithms to improve performance and prevent overfitting.

1. Image augmentation to account for small dataset (horizontal flipping, blurring and stretching)
2. Transfer Learning on VGG16 model to improve performance
3. Cached and prefetched dataset to load images directly from memory during training
4. Early stopping with `min_delta=0.0001` to prevent overfitting
5. Batch Normalization layers to speed up training
6. Utilized a Kaggle GPU to train the model quickly (under 3 minutes)

I also plotted a confusion matrix with Scikit and Seaborn to analyse the F1, precision and recall scores on the test dataset. I intend to use this CNN to classify chess pieces that are cropped from real 3D images of chessboards.
