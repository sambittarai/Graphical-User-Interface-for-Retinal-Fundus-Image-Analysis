import numpy as np
from skimage.measure import label   

def getLargestCC(segmentation):
	# Postprocessing
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    largestCC = np.where(largestCC == True, 1, 0)
    return largestCC

# LCC = getLargestCC(final_prediction)
# LCC.shape