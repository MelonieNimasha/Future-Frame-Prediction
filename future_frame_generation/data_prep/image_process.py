from PIL import Image
import os

def pixel_wise_difference(image1_path, image2_path):
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Ensure the images have the same size
    if image1.size != image2.size:
        raise ValueError("Images must have the same size.")

    # Get the pixel data for both images
    pixels1 = image1.load()
    pixels2 = image2.load()

    # Create a new image to store the pixel-wise difference
    diff_image = Image.new('RGB', image1.size)
    pixels_diff = diff_image.load()

    # Calculate the pixel-wise difference
    for x in range(image1.width):
        for y in range(image1.height):
            r_diff = 2*abs(pixels1[x, y][0] - pixels2[x, y][0])
            g_diff = 2*abs(pixels1[x, y][1] - pixels2[x, y][1])
            b_diff = 2*abs(pixels1[x, y][2] - pixels2[x, y][2])
            pixels_diff[x, y] = (r_diff, g_diff, b_diff)

    return diff_image

def process_images(input_folder1, input_folder2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image filenames from the first input folder
    image_filenames = os.listdir(input_folder1)

    for filename in image_filenames:
        # Build the full paths to the images in both input folders
        image1_path = os.path.join(input_folder1, filename)
        image2_path = os.path.join(input_folder2, filename)

        # Perform pixel-wise difference and get the resulting image
        diff_image = pixel_wise_difference(image1_path, image2_path)

        # Save the resulting image to the output folder with the same name
        output_path = os.path.join(output_folder, filename)
        diff_image.save(output_path)

if __name__ == "__main__":
    input_folder1 = "previous_frames"
    input_folder2 = "prev_previous_frames"
    output_folder = "processed_frames"

    process_images(input_folder1, input_folder2, output_folder)