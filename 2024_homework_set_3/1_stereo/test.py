import numpy as np
from scipy.ndimage import uniform_filter

def compute_matching_cost(left_img, right_img, max_disparity, block_size):
    rows, cols = left_img.shape
    cost_volume = np.zeros((rows, cols, max_disparity), dtype=np.float32)
    
    half_block_size = block_size // 2
    
    for d in range(max_disparity):
        shifted_right = np.roll(right_img, d, axis=1)
        sad = np.abs(left_img - shifted_right)
        sad[:, :d] = 255  # 处理边界
        cost_volume[:, :, d] = uniform_filter(sad, size=block_size)
    
    return cost_volume

def aggregate_costs(cost_volume, P1, P2):
    rows, cols, max_disparity = cost_volume.shape
    aggregated_costs = np.zeros_like(cost_volume)
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 4个方向
    
    for dr, dc in directions:
        cost_dir = np.zeros_like(cost_volume)
        
        for r in range(rows):
            for c in range(cols):
                for d in range(max_disparity):
                    if r - dr >= 0 and c - dc >= 0 and c - dc < cols:
                        prev_costs = cost_dir[r-dr, c-dc, :]
                        min_prev_cost = np.min(prev_costs)
                        penalty = P1 if d == 0 else P2
                        cost_dir[r, c, d] = cost_volume[r, c, d] + min_prev_cost + penalty
                    else:
                        cost_dir[r, c, d] = cost_volume[r, c, d]
                        
        aggregated_costs += cost_dir
    
    return aggregated_costs

def compute_disparity(aggregated_costs):
    disparity_map = np.argmin(aggregated_costs, axis=2)
    return disparity_map

def sgbm(left_img, right_img, max_disparity, block_size, P1=10, P2=100):
    cost_volume = compute_matching_cost(left_img, right_img, max_disparity, block_size)
    aggregated_costs = aggregate_costs(cost_volume, P1, P2)
    disparity_map = compute_disparity(aggregated_costs)
    return disparity_map

# 读取图片
import cv2
left_img = cv2.imread('data/left/000007.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('data/right/000007.png', cv2.IMREAD_GRAYSCALE)

max_disparity = 16
block_size = 5
P1 = 10  # 小的惩罚
P2 = 100  # 大的惩罚

disparity_map = sgbm(left_img, right_img, max_disparity, block_size, P1, P2)
# 用matlibplot显示视差图
import matplotlib.pyplot as plt
plt.imshow(disparity_map, cmap='jet')
plt.colorbar()
plt.show()
