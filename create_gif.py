import argparse
import os

import imageio

def create_gif(folder_path, output_path):

    # List all PNG files in the folder
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

    png_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    # Create a list to store the frames of the GIF
    frames = []

    # Iterate over each PNG file
    for png_file in png_files:
        # Open the PNG file
        image = imageio.v2.imread(os.path.join(folder_path, png_file))
        
        if (image.shape[0] != 480):
            print(image.shape)
            print(png_file)
            continue

        assert image.shape[0] == 480
        assert image.shape[1] == 640
        assert image.shape[2] == 4
        
        # Append the image to the frames list
        frames.append(image)

    # Save the frames as a GIF
    imageio.mimsave(output_path, frames, format='GIF', duration=0.015)
    # imageio.mimwrite(output_path, frames, format='FFMPEG', fps=30)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Convert PNG files to GIF')
    parser.add_argument('folder', help='Path to the folder containing PNG files')
    parser.add_argument('output', help='Path to the output GIF file')
    args = parser.parse_args()

    # Get the folder path and output path from command line arguments
    folder_path = args.folder
    output_path = args.output

    create_gif(folder_path, output_path)


