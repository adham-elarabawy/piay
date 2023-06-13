from PIL import Image
import math
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from PIL import Image

def resize_and_center_crop(image, res=512):
    width, height = image.size
    aspect_ratio = width / height

    if width < height:
        new_width = res
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = res
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    left = (new_width - res) // 2
    top = (new_height - res) // 2
    right = left + res
    bottom = top + res

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

def get_resolution_histogram(images, bins=50):
    resolutions = []
    for image in images:
        width, height = image.size
        resolutions.append((width, height))
    
    width_values, height_values = zip(*resolutions)
    
    plt.figure(figsize=(10, 6))
    plt.hist(width_values, bins=bins, alpha=0.5, label='Width')
    plt.hist(height_values, bins=bins, alpha=0.5, label='Height')
    plt.xlabel('Resolution')
    plt.ylabel('Frequency')
    plt.title('Image Resolution Histogram')
    plt.legend()
    plt.show()

def filter_img_res(image, threshold_width, threshold_height):
    if image:
        width, height = image.size
        if width > threshold_width and height > threshold_height:
            return image
    return None

def crop_center(image, width, height):
    image_width, image_height = image.size
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    right = left + width
    bottom = top + height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except (requests.exceptions.RequestException, IOError):
        return None

def image_grid(images, grid_width, grid_height, res=512):
    # Calculate the total number of images and the grid size
    num_images = len(images)
    grid_size = grid_width * grid_height
    
    # If there are more images than grid size, raise an error
    if num_images > grid_size:
        raise ValueError("The number of images exceeds the grid size.")
    
    # Calculate the dimensions of each grid cell
    cell_width = res #math.ceil(max(image.width for image in images))
    cell_height = res #math.ceil(max(image.height for image in images))
    
    # Create a blank canvas for the grid
    grid_canvas = Image.new('RGB', (grid_width * cell_width, grid_height * cell_height))
    
    # Iterate over each image and place it in the grid
    for i, image in enumerate(images):
        # Calculate the coordinates for the current grid cell
        x = (i % grid_width) * cell_width
        y = (i // grid_width) * cell_height
        
        # Resize the image to fit the grid cell size
        resized_image = image.resize((cell_width, cell_height))
        
        # Paste the resized image onto the grid canvas
        grid_canvas.paste(resized_image, (x, y))
    
    return grid_canvas


