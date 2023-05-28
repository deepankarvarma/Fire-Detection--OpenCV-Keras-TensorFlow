# Fire Detection Model

This repository contains Python code for generating a fire detection model and using it to detect fire from user-uploaded images. The model architecture is defined as follows:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Dataset

The dataset used for training and evaluation can be downloaded from Kaggle: [Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset). It consists of two categories: `fire_images` and `non_fire_images`, containing labeled images for fire and non-fire classes.

## Dependencies

To run the code in this repository, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required packages using `pip`:

```shell
pip install tensorflow keras numpy matplotlib
```

## Usage

1. Clone this repository to your local machine:

```shell
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the Fire Dataset from the provided link and place it in the appropriate directory.

3. Use the provided code to train the fire detection model.

4. Run the script to detect fire from an uploaded image:

```shell
python predict_image.py --image path/to/your/image.jpg
```

Make sure to replace `path/to/your/image.jpg` with the actual path to your desired image file.

## Results

The trained fire detection model can accurately classify images as containing fire or not. You can modify the code and experiment with different architectures or hyperparameters to potentially improve the performance.

## Acknowledgments

- The Fire Dataset used in this project was sourced from Kaggle: [Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset).

## License

This project is licensed under the [MIT License](LICENSE).

