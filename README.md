# imutil

This is a simple library for displaying images.

## Usage

````
    pixels = np.random.normal(size=(100,100))
    imutil.show(pixels)  # Outputs a JPG monochrome image


    pixels = np.random.normal(size=(100,100,3))
    imutil.show(pixels)  # Outputs a JPG color image


    pixels = np.random.normal(size=(16,100,100,3))
    imutil.show(pixels)  # Outputs a JPG color image of a 4 by 4 grid
````

## Displaying In-Terminal

If you use iTerm2 with `imgcat`, set the environment variable `IMUTIL_SHOW=1` to display output images in your terminal.

