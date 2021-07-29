"""
File: stanCodoshop.py
----------------------------------------------
SC101_Assignment3
Adapted from Nick Parlante's
Ghost assignment by Jerry Liao.

-----------------------------------------------

TODO: The program will first load all the images from certain directory. Then remove the pedestrians in the image
        by calculating the best pixel among several images. The best pixel means the shortest distance between the RGB of
        one pixel and the average RGB of pixels of those images.
        (Think of RGB as XYZ so that we can calculate the distance.)
"""

import os
import sys
from simpleimage import SimpleImage


def get_pixel_dist(pixel, red, green, blue):
    """
    Returns the color distance between pixel and mean RGB value

    Input:
        pixel (Pixel): pixel with RGB values to be compared
        red (int): average red value across all images
        green (int): average green value across all images
        blue (int): average blue value across all images

    Returns:
        dist (float): color distance between red, green, and blue pixel values

    """
    dist = ((red - pixel.red) ** 2 + (green - pixel.green) ** 2 + (blue - pixel.blue) ** 2) ** 0.5
    return dist


def get_average(pixels):
    """
    Given a list of pixels, finds the average red, blue, and green values

    Input:
        pixels (List[Pixel]): list of pixels to be averaged
    Returns:
        rgb (List[int]): list of average red, green, blue values across pixels respectively

    Assumes you are returning in the order: [red, green, blue]

    """
    red = 0
    green = 0
    blue = 0
    rgb = []
    for pixel in pixels:
        red += pixel.red
        green += pixel.green
        blue += pixel.blue

    avg_red = red // len(pixels)
    avg_green = green // len(pixels)
    avg_blue = blue // len(pixels)

    rgb.append(avg_red)
    rgb.append(avg_green)
    rgb.append(avg_blue)

    return rgb


def get_best_pixel(pixels):
    """
    Given a list of pixels, returns the pixel with the smallest
    distance from the average red, green, and blue values across all pixels.

    Input:
        pixels (List[Pixel]): list of pixels to be averaged and compared
    Returns:
        best (Pixel): pixel closest to RGB averages

    """

    pixel_be_compared = {}
    lis_avg_pixel = get_average(pixels)

    for pixel in pixels:
        dist = get_pixel_dist(pixel, lis_avg_pixel[0], lis_avg_pixel[1], lis_avg_pixel[2])
        # key = pixel ; value = dist
        pixel_be_compared[pixel] = dist

    best = min(pixel_be_compared, key=pixel_be_compared.get)
    return best


def solve(images):
    """
    Given a list of image objects, compute and display a Ghost solution image
    based on these images. There will be at least 3 images and they will all
    be the same size.

    Input:
        images (List[SimpleImage]): list of images to be processed
    """
    width = images[0].width
    height = images[0].height
    result = SimpleImage.blank(width, height)
    ######## YOUR CODE STARTS HERE #########
    # Write code to populate image and create the 'ghost' effect

    for y in range(result.height):
        for x in range(result.width):

            prepare_pixel = list(image.get_pixel(x, y) for image in images)
            best = get_best_pixel(prepare_pixel)

            new_pixel = result.get_pixel(x, y)
            new_pixel.red = best.red
            new_pixel.green = best.green
            new_pixel.blue = best.blue

    ######## YOUR CODE ENDS HERE ###########
    print("Displaying image!")
    result.show()


def jpgs_in_dir(dir):
    """
    (provided, DO NOT MODIFY)
    Given the name of a directory, returns a list of the .jpg filenames
    within it.

    Input:
        dir (string): name of directory
    Returns:
        filenames(List[string]): names of jpg files in directory
    """
    filenames = []
    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            filenames.append(os.path.join(dir, filename))
    return filenames


def load_images(dir):
    """
    (provided, DO NOT MODIFY)
    Given a directory name, reads all the .jpg files within it into memory and
    returns them in a list. Prints the filenames out as it goes.

    Input:
        dir (string): name of directory
    Returns:
        images (List[SimpleImages]): list of images in directory
    """
    images = []
    jpgs = jpgs_in_dir(dir)
    for filename in jpgs:
        print("Loading", filename)
        image = SimpleImage(filename)
        images.append(image)
    return images


def main():
    # (provided, DO NOT MODIFY)
    args = sys.argv[1:]
    # We just take 1 argument, the folder containing all the images.
    # The load_images() capability is provided above.
    images = load_images(args[0])
    solve(images)


if __name__ == '__main__':
    main()
