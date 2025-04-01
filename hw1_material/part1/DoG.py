import cv2
import numpy as np

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_gaussian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        gaussian_images = []
        dog_images = []

        # Step 1: Generate the first octave
        first_octave = [image] + [cv2.GaussianBlur(image, (0, 0), self.sigma**i) for i in range(1,self.num_gaussian_images_per_octave)]

        # Step 2: Down-sample the last image
        DSImage = cv2.resize(first_octave[-1], 
                             (image.shape[1]//2, image.shape[0]//2), 
                             interpolation = cv2.INTER_NEAREST)

        # Step 3: Generate the second octave
        second_octave = [DSImage] + [cv2.GaussianBlur(DSImage, (0, 0), self.sigma**i) for i in range(1,self.num_gaussian_images_per_octave)]

        # Combine into a list
        gaussian_images = [first_octave, second_octave]

        # Step 4: Compute DoG images and save them
        for octave_idx, octave in enumerate(gaussian_images):
            dog_octave = []
            for i in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(octave[i ], octave[i + 1])
                dog_octave.append(dog)

                # Normalize and save DoG images
                M, m = max(dog.flatten()), min(dog.flatten())
                norm = ((dog - m) * 255) / (M - m)  # Normalize to 0-255 for image display
                #cv2.imwrite(f'DoG_octave_{octave_idx + 1}_{i + 1}.png', norm)

            dog_images.append(dog_octave)


        # Step 5: Find keypoints
        keypoints = []
        # Process each octave
        for octave_idx, octave in enumerate(dog_images):
            scale = 2**octave_idx  # Scale factor for this octave
            # Iterate over the 2nd and 3rd images of the octave (skip first and last)
            for i in range(1, len(octave)-1):
                for y in range(1, octave[i].shape[0] - 2):
                    for x in range(1, octave[i].shape[1] - 2):
                        # Check if the pixel is a local maximum or minimum
                        pixel = octave[i][y, x]
                        
                        if np.abs(pixel) <= self.threshold:
                            continue
                        #rint(f'Pixel value: {pixel}')
                        # Get the 26 neighbors
                        neighbors = np.array([octave[i - 1][y - 1:y + 2, x - 1:x + 2],
                                            octave[i][y - 1:y + 2, x - 1:x + 2],
                                            octave[i + 1][y - 1:y + 2, x - 1:x + 2]])
                        neighbors = neighbors.reshape(-1)
                        #print(f'Neighbors: {neighbors}')
                        # Check if the pixel is a local maximum or minimum
                        if pixel >= np.max(neighbors) or pixel <= np.min(neighbors):
                            # Scale coordinates according to the octave
                            keypoints.append([y * scale, x * scale])


        keypoints = np.unique(keypoints, axis=0)

        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        print(len(keypoints))
        
        return keypoints