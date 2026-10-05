# Fashion Product Image Classification with CNNs

Automatically classifying clothing product images into 10 categories with convolutional neural networks, to reduce manual product tagging for an online retailer. Part of the IBM Machine Learning Professional Certificate (Deep Learning).

## Data

- **Source:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando product images), loaded with `keras.datasets.fashion_mnist`
- **Size:** 70,000 greyscale images at 28 × 28 pixels (60,000 train, 10,000 test)
- **Classes:** 10 balanced categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Preparation:** pixels scaled to 0–1, channel dimension added, 10% of training data held out for validation

## Models

| Model | Parameters | Epochs | Final train acc. | Best val. acc. | **Test acc.** | Test loss |
|---|---|---|---|---|---|---|
| Dense neural network | 101,770 | 15 | 91.9% | 88.8% | 88.3% | 0.346 |
| Basic CNN (2 × Conv + MaxPool) | 421,642 | 15 | 97.2% | 92.2% | 90.9% | 0.314 |
| Regularised CNN (+ BatchNorm, Dropout, EarlyStopping) | 422,026 | 15 | 91.4% | 91.6% | 91.0% | 0.269 |
| **Regularised CNN + learning-rate schedule** | 422,026 | 38 | 94.1% | 93.3% | **93.0%** | **0.225** |

The final model adds `ReduceLROnPlateau` and longer early-stopping patience. It has the highest accuracy, the lowest loss and the smallest train/validation gap.

## Key findings

- **About 40% fewer errors** than the dense baseline (error rate 11.7% → 7.0%).
- Trouser, Sandal, Bag, Sneaker and Ankle boot reach about 97–99% recall.
- **Shirt is the hardest class** (77.1% recall). It is mostly confused with T-shirt/top (87), Coat (56) and Pullover (52).
- **Confidence-based automation:** 93.5% of products are predicted with at least 70% confidence, and those predictions are 95.9% accurate. A retailer could auto-tag those and send the remaining 6.5% to human review.

## Limitations and next steps

Small greyscale images compared with real product photos, confusion between upper-body garments, a single validation split, and no augmentation. Next steps: data augmentation, transfer learning (MobileNet, ResNet), higher-resolution colour images, and a specialised classifier for tops.

## Files

- `fashion_mnist_cnn.ipynb`: full notebook (Google Colab, GPU)
- `Fashion_Product_Classification_Report.pdf`: stakeholder report

## Tools

Python, TensorFlow / Keras, NumPy, pandas, scikit-learn, Matplotlib
