import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to contact TAs

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # These are placeholders - replace with the coordinates of your interest points!

    # 设置参数
    sigma = 1.0
    k = 0.06
    threshold = 0.1
    min_distance = 10

    # 从main.py中可以知道传入的图片都是灰度图，可以直接计算Ix, Iy,求得海森矩阵
    Ix = filters.sobel_v(image=image)
    Iy = filters.sobel_h(image=image)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # 将高斯滤波器用于上面算子
    Ixx = filters.gaussian(Ixx, sigma=sigma)
    Iyy = filters.gaussian(Iyy, sigma=sigma)
    Ixy = filters.gaussian(Ixy, sigma=sigma)

    # 计算每个像素的 R= det(H) - k(trace(H))²。det(H)表示矩阵H的行列式，trace表示矩阵H的迹。通常k的取值范围为[0.04,0.16]。
    detH = (Ixx * Iyy) - (Ixy ** 2)
    traceH = Ixx + Iyy
    R = detH - k * ((traceH) ** 2)

    # 满足 R>=max® * th 的像素点即为角点。th常取0.1，不知道peak_local_max的threshold_rel是不是就是在做这个
    # 参数中的feature_width好像是没用的，又或者说他与min_distance有关? 我感觉它只是在get_features中会用到
    points = feature.peak_local_max(R, min_distance=min_distance, threshold_rel=threshold)
    
    xs = points[:, 1]
    ys = points[:, 0]

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to contact TAs

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # This is a placeholder - replace this with your features!
    # assert feature_width % 4 == 0
    
    
    h = image.shape[0]
    w = image.shape[1]

    offset = feature_width // 2
    num_points = x.shape[0]

    # 计算索引范围
    x_start = x - offset
    x_stop = x + offset
    y_start = y - offset
    y_stop = y + offset

    # Compute padding parameters
    x_min = x_start.min()
    x_max = x_stop.max()
    y_min = y_start.min()
    y_max = y_stop.max()

    x_pad = [0, 0]
    y_pad = [0, 0]
    if x_min < 0:
        x_pad[0] = -x_min
    if y_min < 0:
        y_pad[0] = -y_min
    if x_max - w >= 0:
        x_pad[1] = x_max - w + 1
    if y_max - h >= 0:
        y_pad[1] = y_max - h + 1

    x_start += x_pad[0]
    x_stop += x_pad[0]
    y_start += y_pad[0]
    y_stop += y_pad[0]

    # 对图片进行pudding
    image = np.pad(image, [y_pad, x_pad], mode="constant")

    # 对每个维度窗口下建立索引
    cell_size = 4
    num_blocks = feature_width // cell_size

    x_idx = np.array([np.arange(start, stop) for start, stop in zip(x_start, x_stop)])
    y_idx = np.array([np.arange(start, stop) for start, stop in zip(y_start, y_stop)])

    # 转置之前 ：(num_blocks, num_points, cell_size)
    x_idx = np.array(np.split(x_idx, num_blocks, axis=1))
    y_idx = np.array(np.split(y_idx, num_blocks, axis=1))

    # 转置之后：(num_points, num_blocks, cell_size)
    x_idx = x_idx.transpose([1, 0, 2])
    y_idx = y_idx.transpose([1, 0, 2])

    # 对窗口中的每个像素建立索引
    x_idx = np.tile(np.tile(x_idx, cell_size), [1, num_blocks, 1]).flatten()
    y_idx = np.tile(np.repeat(y_idx, cell_size, axis=2), num_blocks).flatten()

    # 计算偏导数
    partial_x = filters.sobel_h(image)
    partial_y = filters.sobel_v(image)
    # 计算梯度模长
    magnitude = np.sqrt(partial_x * partial_x + partial_y * partial_y)
    # 计算梯度方向
    orientation = np.arctan2(partial_y, partial_x) + np.pi
    # 梯度近似为最近的角度
    orientation = np.mod(np.round(orientation / (2.0 * np.pi) * 8.0), 8)
    orientation = orientation.astype(np.int32)
    # 对梯度做高斯平滑
    magnitude = filters.gaussian(magnitude, sigma=offset)

    # 所有的块数据转为1维
    magnitude_in_pixels = magnitude[y_idx, x_idx]
    orientation_in_pixels = orientation[y_idx, x_idx]

    # 转为数组 (num_patches, cell_size, cell_size)
    magnitude_in_cells = magnitude_in_pixels.reshape((-1, cell_size * cell_size))
    orientation_in_cells = orientation_in_pixels.reshape((-1, cell_size * cell_size))

    # 对每个cel计算梯度方向加权和
    features = np.array(list(
        map(lambda array, weight: np.bincount(array, weight, minlength=8), orientation_in_cells, magnitude_in_cells)))

    # 每一行都是角点对应的一个特征向量
    features = features.reshape((num_points, -1))
    features = features / np.linalg.norm(features, axis=-1).reshape((-1, 1))
    features[features >= 0.2] = 0.2
    features = features / np.linalg.norm(features, axis=-1).reshape((-1, 1))

    return features

def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # These are placeholders - replace with your matches and confidences!

    # matches = np.zeros((1, 2))
    # confidences = np.zeros(1)

    assert im1_features.shape[1] == im2_features.shape[1]
    # 设置confidence
    confidence = 0.95

    matches = []
    confidences = []

    for i in range(im1_features.shape[0]):
        distances = np.sqrt(
            ((im1_features[i, :] - im2_features) ** 2).sum(axis=1))
        indexes = np.argsort(distances)
        min_distance = distances[indexes[0]]
        second_min_distance = distances[indexes[1]]

        ratio = min_distance / second_min_distance if second_min_distance != 0 else 0
        if ratio < confidence:
            matches.append([i, indexes[0]])
            confidences.append(1 - ratio)

    matches = np.asarray(matches)
    confidences = np.asarray(confidences)

    return matches, confidences
