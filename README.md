# Image Compression

This project implements basic image compression techniques as part of the CS229 (Autumn 2018) Problem Set 3. The main script, `image-compression.py`, demonstrates how to compress and reconstruct images using k-means by grouping pixel colors into clusters

## Files

- `image-compression.py`: Main script for performing image compression and reconstruction.
- `peppers-large.tiff`: Example input image used for compression.
- `output/compressed.jpg`: Example output of the compressed image.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- PIL (Pillow)

You can install the required packages using:

```bash
pip install numpy matplotlib pillow
```

## Usage

1. Place your input image (e.g., `peppers-large.tiff`) in the project directory.
2. Run the main script:

```bash
python image-compression.py
```

3. The script will read the input image, perform compression, and save the result to the `output/` directory as `compressed.jpg`.

## Output

- The compressed image will be saved in the `output/` folder.
- The script may also display visualizations comparing the original and compressed images.

## References

- [CS229: Machine Learning (Stanford University)](http://cs229.stanford.edu/)
- [k-means (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)
