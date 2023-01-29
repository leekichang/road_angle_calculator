import os
import cv2
import numpy as np

import requests
from utils import API_KEY

def get_map_image(latitude:str, longitude:str, size:tuple=(500, 500), zoom:int=20)->bytes:
    """
    Args
    latitude, logitude: 위도 경도 값
    size: 이미지 크기를 알려주는 tuple. (H,W)
    zoom: zoom 정도 기본값은 20
    return: 지도 image의 byte data
    """
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=roadmap&style=feature:all|element:labels|visibility:off&key={API_KEY}"
    response = requests.get(url)
    return response.content

def bytes2img(data:bytes)->np.ndarray:
    """
    Args
    data: image byte 데이터
    return: cv2 기반 image 데이터
    """
    data_ = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(data_, cv2.IMREAD_COLOR)
    return img

def find_closest_nonzero_point(img:np.ndarray)->tuple:
    """
    Args
    img: edge detection을 돌린 Grayscale image
    return: 이미지의 중앙에서 가장 가까운 edge point의 좌표 
    """
    H, W = img.shape
    center_x, center_y = W // 2, H // 2
    # Find the coordinates of all non-zero pixels
    nonzero_indices = np.nonzero(img)
    nonzero_coords = list(zip(nonzero_indices[0], nonzero_indices[1]))
    # Calculate the distance of each non-zero pixel to the center
    distances = [np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in nonzero_coords]
    # Find the index of the non-zero pixel with the minimum distance
    closest_index = np.argmin(distances)
    # Return the coordinates of the closest non-zero pixel
    return nonzero_coords[closest_index]

def visualize_points(img, point, kernel_size)->np.ndarray:
    # Get the coordinates of the point
    y, x = point
    # Create a window around the point
    window = img[y-kernel_size//2:y+kernel_size//2, x-kernel_size//2:x+kernel_size//2]
    # Find the non-zero points in the window
    non_zero_points = cv2.findNonZero(window)
    # Create a copy of the original image
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    # Draw circles around the non-zero points
    for p in non_zero_points:
        p = p[0]
        cv2.circle(img_copy, (p[0]+x-kernel_size//2, p[1]+y-kernel_size//2), 1, (0, 0, 255), -1)
    cv2.rectangle(img_copy, (x-kernel_size//2, y-kernel_size//2), (x+kernel_size//2, y+kernel_size//2), (0, 255, 0), 2)
    return img_copy

def visualize_point(img:np.ndarray, point:tuple)->np.ndarray:
    """
    Args
    img: edge detection을 수행한 grayscale image
    point: 시각화하고자 하는 점
    return: None
    """
    y, x = point
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(rgb_img, (x, y), 3, (0, 0, 255), -1)
    return rgb_img
    # cv2.imshow(f"coord=(y={y}, x={x})", rgb_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def find_nonzero_in_window(img:np.ndarray, point:tuple, kernel_size:int=5)->list:
    """
    Args
    img: edge detection을 수행한 grayscale image
    point: (y, x) 순서로 배열된 tuple
    kernel_size: window의 크기
    return: 0이 아닌 값이 포함된 점들의 좌표를 담은 list
    특정 point에서 kernel size 내부에서 nonzero 값을 가지는 모든 점들을 list에 저장하여 반환하는 함수
    img가 edge detection을 수행한 이미지이므로 edge들의 좌표
    """
    
    y, x = point
    window_x = x - kernel_size // 2
    window_y = y - kernel_size // 2
    
    window = img[window_y:window_y+kernel_size, window_x:window_x+kernel_size]

    nonzero_points = np.transpose(np.nonzero(window))
    
    nonzero_points[:, 0] += window_y
    nonzero_points[:, 1] += window_x
    
    return nonzero_points

def find_line_from_points(points_:np.ndarray)->tuple:
    """
    Args
    points_: [[y1, x1], [y2, x2], [y3, x3]...] 형태의 좌표를 담은 array
    return: 직선의 기울기와 y절편
    주어진 점들에 대해서 least square error를 가지는 직선의 기울기와 y절편을 구하는 함수
    """
    if len(points_) == 0:
        return
    points = np.array([[x,y] for (y,x) in points_])
    # fit a line to the points using cv2.fitLine()
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # calculate the slope and y-intercept of the line
    slope = vy / vx
    intercept = y - slope * x
    return slope, intercept

def visualize_line(img:np.array,
                   slope:float,
                   intercept:float,
                   color:tuple=(0,0,255),
                   thickness:int=2)->np.ndarray:
    """
    Args
    img: edge detection을 수행한 grayscale image
    slope: 표시할 직선의 기울기
    intercept: 표시할 직선의 y절편
    color: 표시할 직선의 색상
    tickness: 표시할 직선의 두께
    return: 직선의 기울기와 y절편
    주어진 점들에 대해서 least square error를 가지는 직선의 기울기와 y절편을 구하는 함수
    """
    # get the image shape
    rows, cols = img.shape[:2]
    # find two points on the line that correspond to the edges of the image
    left_x = int((0 - intercept) / slope)
    right_x = int((rows - intercept) / slope)
    start = (left_x, 0)
    end = (right_x, rows)
    # draw the line on the image using cv2.line()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.line(img, start, end, color, thickness)
    return img
    

if __name__ == '__main__':

    latitude = "37.386363"
    longitude = "126.644891"

    if os.path.exists(f'imgs/{latitude}_{longitude}.png'):
        img = cv2.imread(f'imgs/{latitude}_{longitude}.png')
    else:
        data = get_map_image(latitude, longitude)
        img = bytes2img(data)
        cv2.imwrite(f'imgs/{latitude}_{longitude}.png', img)

    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('detected edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    point = find_closest_nonzero_point(edges)
    tmp = visualize_point(edges, point)
    
    cv2.imshow(f"coord=(y={point[0]}, x={point[1]})", tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    tmp = visualize_points(edges, point, 20)
    cv2.imshow("image", tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points = find_nonzero_in_window(edges, point, kernel_size=20)
    slope, intercept = find_line_from_points(points)
    print(f"The angle of road is {np.rad2deg(np.arctan(slope))[0]}°")
    
    tmp = visualize_line(edges, slope, intercept)
    cv2.imshow('fitted line', tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

