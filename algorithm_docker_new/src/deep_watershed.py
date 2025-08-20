from typing import Optional
import numpy as np
from skimage.segmentation import watershed
from scipy.ndimage import label, find_objects
from skimage.measure import label, regionprops


def deep_watershed_with_voting(deep_watershed_basins: np.array, multiclass_segmentation: np.array, binary_mask: Optional[np.array] = None,
                               markers: Optional[np.array] = None, instance_ground_truth: Optional[np.array] = None,
                               seed_distance_treshold: float = 0.5, calculate_metrics: bool = False):
    
    #check required inputs for deep watershed algorithm
    if markers is None:
        markers = deep_watershed_basins.copy()
        markers = np.where(markers > seed_distance_treshold, 1, 0)
    if binary_mask is None:
        binary_mask = multiclass_segmentation.copy()
        binary_mask = np.where(binary_mask >= 1, 1, 0)
    
    #label seeds 
    instances = label(markers, connectivity=3, return_num=False)

    #apply watershed algorithm based on negative distance using instances as seeds and masked by binary segmentation
    instance_masks = watershed(-deep_watershed_basins, instances, mask=binary_mask)

    #get instance masks and bboxes (regionprops) and perform majority voting to assign classes to instances
    output = np.zeros_like(instance_masks)
    instance_masks_props = regionprops(instance_masks)

    voting_instances = [] 
    
    #MAJORITY VOTING
    for idx, i in enumerate(instance_masks_props):        
        #get tooth instance voxels based on region of interest fr                                                                                                                                                                      om original model output
        pred_instance  = multiclass_segmentation[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
        #get class with the most votes (ignore background class)
        votes_pred = np.bincount(pred_instance)
        
        majority_class_pred = 0
        if len(votes_pred)>1:
            majority_class_pred = np.argmax(votes_pred[1:])+1

        #relabel instance voxels based on winner
        voting_instances.append(majority_class_pred)
        output[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image] = majority_class_pred

    return output

def deep_watershed_with_voting_optimized(
    deep_watershed_basins: np.ndarray,
    multiclass_segmentation: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    markers: Optional[np.ndarray] = None,
    seed_distance_threshold: float = 0.5,
) -> np.ndarray:
    """
    Optimized deep watershed with majority voting on 3D scans.
    Produces identical results to original function.

    Parameters
    ----------
    deep_watershed_basins : np.ndarray
        Distance map or watershed basins.
    multiclass_segmentation : np.ndarray
        Original predicted multiclass segmentation.
    binary_mask : np.ndarray, optional
        Binary mask for watershed, defaults to multiclass_segmentation > 0.
    markers : np.ndarray, optional
        Seed markers for watershed, defaults to deep_watershed_basins > seed_distance_threshold.
    seed_distance_threshold : float
        Threshold to create markers if markers not provided.

    Returns
    -------
    np.ndarray
        Instance segmentation with majority class voting.
    """

    # Prepare markers
    if markers is None:
        markers = (deep_watershed_basins > seed_distance_threshold).astype(np.uint8)

    # Prepare binary mask
    if binary_mask is None:
        binary_mask = (multiclass_segmentation >= 1).astype(np.uint8)

    # Label seeds
    instances = label(markers, connectivity=3, return_num=False)

    # Apply watershed
    instance_masks = watershed(-deep_watershed_basins, instances, mask=binary_mask)

    # Prepare output
    output = np.zeros_like(instance_masks, dtype=np.uint8)

    # Get slices of all instances
    slices = find_objects(instance_masks)

    for label_idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        # Mask of current instance
        instance_mask = (instance_masks[slc] == label_idx)

        if not instance_mask.any():
            continue

        # Extract corresponding voxels in multiclass segmentation
        pred_instance = multiclass_segmentation[slc][instance_mask]

        # Majority voting ignoring background
        votes_pred = np.bincount(pred_instance)
        majority_class_pred = 0
        if len(votes_pred) > 1:
            majority_class_pred = np.argmax(votes_pred[1:]) + 1

        # Assign to output directly
        output[slc][instance_mask] = majority_class_pred

    return output