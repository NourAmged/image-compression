from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os



def init_centroids(num_clusters, image):
    
    H, W, C = image.shape    
    
    centroids = np.zeros((num_clusters, C))
    
    for cluster in range(num_clusters):
        
        rand_row = np.random.randint(0, H)
        rand_col = np.random.randint(0, W)
        
        centroids[cluster] = image[rand_row, rand_col]
        
    return centroids


def update_centroids(centroids, image, max_iter = 30, eps = 2):

    H, W, _ = image.shape
    
    K = centroids.shape[0]
    i = 0

    prev_centroids = None
    current_centroids = centroids.copy()
    
    while i < max_iter and (prev_centroids is None or (np.linalg.norm(prev_centroids - current_centroids) > eps)):
        
        print("Iteration: ", i + 1)
        
        centroid_to_pixels = {k: [] for k in range(K)}
        
        for row in range(H):
            for col in range(W):
                pixel = image[row, col]
                nearest_centroid_idx = np.argmin(np.linalg.norm(current_centroids - pixel, axis = 1))
                centroid_to_pixels[nearest_centroid_idx].append(pixel)

   
        prev_centroids = current_centroids.copy()

        for k in range(K):
            if centroid_to_pixels[k]:
                current_centroids[k] = np.mean(centroid_to_pixels[k], axis = 0)
        
        print(np.linalg.norm(prev_centroids - current_centroids))      
        i += 1
    return current_centroids
                
                
                
def update_image(image, centroids):
    
    image_copy = image.copy()

    H, W, _ = image_copy.shape
    
    for row in range(H):
        for col in range(W):
            nearest_centroid = np.argmin(np.linalg.norm(centroids - image_copy[row, col], axis = 1))
            image_copy[row, col] = centroids[nearest_centroid]
            
    return image_copy        
    
    
    
    
def main(image_path, num_clusters = 16, max_iter = 30, eps = 2):

    img = Image.open(image_path)
    img = img.convert("RGB")
    
    img = np.asarray(img)
    
    centroids = init_centroids(num_clusters, image = img)

    centroids = update_centroids(centroids = centroids, image = img, max_iter = max_iter, eps = eps)

    image = update_image(image = img, centroids = centroids)

    plt.imshow(image)

    image = Image.fromarray(image)

    try:
        os.makedirs('output')
        print("Directory created successfully.")
    except FileExistsError:
        print("Directory already exists.")


    file_name = "compressed_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image.save(f'output/{file_name}.jpg')

    plt.show()
    

main('peppers-large.tiff')