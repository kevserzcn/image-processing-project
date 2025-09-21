# Image Processing Project

Image processing interface coded with Python using Tkinter GUI. This application provides a comprehensive set of basic image processing functions with an intuitive graphical user interface.

## Features

### Core Image Processing Operations
- **Binary Conversion** - Convert images to binary using adjustable thresholds
- **Canny Edge Detection** - Detect edges with customizable low and high thresholds
- **Image Cropping** - Crop images with precise coordinate controls
- **Color Space Conversions** - Convert between RGB, HSV, Grayscale, and LAB color spaces
- **Noise Operations** - Add various types of noise (Gaussian, Salt & Pepper, Poisson, Speckle)
- **Grayscale Conversion** - Convert color images to grayscale
- **Contrast and Brightness Adjustment** - Fine-tune image appearance

### Advanced Processing
- **Morphological Operations** - Erosion, dilation, opening, closing, gradient, tophat, blackhat
- **Histogram Operations** - Equalization, stretching, and visualization
- **Geometric Transformations** - Rotation (clockwise/counterclockwise) and scaling
- **Zoom Operations** - Zoom in/out with precise control
- **Arithmetic Operations** - Add, subtract, multiply, divide two images
- **Filtering** - Median filter, motion filter, Gaussian blur, bilateral filter
- **Noise Removal** - Multiple denoising methods

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - opencv-python
  - Pillow (PIL)
  - matplotlib
  - tkinter (usually included with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kevserzcn/image-processing-project.git
cd image-processing-project
```

2. Install required packages:
```bash
pip install numpy opencv-python pillow matplotlib
```

3. Run the application:
```bash
python gi.py
```

## Usage

1. **Load an Image**: Click "Görüntü Seç" to select your input image
2. **Choose Operation**: Select from 19 different image processing operations in the side panel
3. **Adjust Parameters**: Use the parameter controls that appear based on your selected operation
4. **Apply**: Click "Uygula" to process the image
5. **Save Result**: Use "İşlenmiş Görüntüyü Kaydet" to save the processed image

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## Interface Language
The interface is in Turkish, with the following key buttons:
- **Görüntü Seç** - Select Image
- **İkinci Görüntü Seç** - Select Second Image (for arithmetic operations)
- **Uygula** - Apply
- **İşlenmiş Görüntüyü Kaydet** - Save Processed Image

## Architecture

The application is built using:
- **Tkinter** for the GUI framework
- **OpenCV** for core image processing algorithms
- **NumPy** for efficient array operations
- **PIL/Pillow** for image display and format handling
- **Matplotlib** for histogram visualization

## Contributing

Feel free to contribute to this project by:
- Adding new image processing operations
- Improving the user interface
- Adding support for more image formats
- Optimizing processing algorithms

## License

This project is open source and available under the MIT License.
