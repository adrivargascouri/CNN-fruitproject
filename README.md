# CNN Project: Automatic Fruit Classification

This project implements a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify images of fruits. The model can identify the fruit class from 150x150 pixel color images and serves as a solid foundation for inventory apps, smart agriculture, supermarkets, education, and more.

---

## Table of Contents

- [Description](#description)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Dataset Visualization](#dataset-visualization)
- [Model Definition and Compilation](#model-definition-and-compilation)
- [Model Training](#model-training)
- [Evaluation and Results](#evaluation-and-results)
- [Usage Example (Inference)](#usage-example-inference)
- [Author and License](#author-and-license)

---

## Description

- **Input:** Image of a fruit (150x150 pixels, 3 RGB channels)
- **Output:** Fruit class label
- **Practical application:** Automates fruit recognition in images.

---

## Technologies Used

- Python 3.x
- TensorFlow (v2.18.0)
- Keras
- NumPy
- Matplotlib
- Google Colab (for running and Google Drive access)
- Dataset: (https://www.kaggle.com/code/etatbak/cnn-fruit-classification/input)
- Dataset : link to the final Dataset i used --> https://drive.google.com/drive/folders/1Cgb9PdnTZ7IS4O-lsujBNTOrBnSvz84I?usp=sharing

---

## Repository Structure

```
cnn_proyect/
│
├── fruits_selected/
│   ├── training/
│   └── testing/
├── cnn_proyect.ipynb
├── README.md
└── ...
```

- `fruits_selected/`: Folder with data divided into `training` and `testing`
- `cnn_proyect.ipynb`: Main notebook with project code

---

## Environment Setup

Install the primary dependencies:

```bash
pip install tensorflow matplotlib numpy
```

---

## Data Preparation

1. **Mount Google Drive and copy dataset:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Ensure the structure:**
    ```
    /content/drive/MyDrive/fruits_selected/
        training/
        testing/
    ```

3. **Verify the contents:**
    ```python
    import os
    ruta_fruits = '/content/drive/MyDrive/fruits_selected'
    print("Files in 'fruits_selected':", os.listdir(ruta_fruits))
    ```

---

## Dataset Visualization

```python
images, labels = next(train_generator)
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    label = np.argmax(labels[i])
    class_name = list(train_generator.class_indices.keys())[label]
    plt.title(class_name)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Model Definition and Compilation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Input(shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Model Training

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

---

## Evaluation and Results

### Test Set Evaluation

```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

### Plotting Accuracy and Loss

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

#### Example Results (update with your metrics):

- **Final Test Accuracy:** Test Accuracy: 0.9574
- **Final Test Loss:** Test Loss: 0.1285
- **Maximum validation accuracy:**
- ![Screenshot 2025-06-21 211331](https://github.com/user-attachments/assets/897d1432-9877-4779-b59c-710e328ae208)


---

## Usage Example (Inference)

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path/to/an/image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_index = np.argmax(prediction)
class_name = list(train_generator.class_indices.keys())[class_index]
print(f"The image is: {class_name}")
```
![Screenshot 2025-06-21 211245](https://github.com/user-attachments/assets/0b990477-2398-43ad-ba4e-dfc92eb03d50)

---

## Author and License

Created by [adrivargascouri](https://github.com/adrivargascouri)

MIT License.

---
