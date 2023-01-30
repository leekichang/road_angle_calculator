from typing import List, Union, Tuple
import requests
import os
import cv2
import shutil
import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import haversine

from calcRoadAngle import *

MAX_WORKERS = 8
API_KEY = "AIzaSyAqm2-G0bi6UGtdb7wCMK8cd50qrZ4q2AQ"

class GeoInfo:
    def __init__(self, latlng:Union[str, Tuple[float]]=(-1, -1), fov=-1, heading=-1, pitch=-1):
        if type(latlng) is GeoInfo:
            # Copy constructor
            lat     = latlng.lat
            lng     = latlng.lng
            fov     = latlng.fov
            heading = latlng.heading
            pitch   = latlng.pitch
        elif type(latlng) is str and latlng.count(",") == 1:
            lat, lng = latlng.split(",")
            lat, lng = lat.strip(), lng.strip()
        elif type(latlng) == tuple:
            lat, lng = latlng
        else:
            raise ValueError("Wrong arguments for GPS info (must be lat,lng format)")
        
        self.lat        = float(lat)
        self.lng        = float(lng)
        self._init_additional_info(fov, heading, pitch)
    
    def update_additional_info(self, fov, heading, pitch):
        self._init_additional_info(fov, heading, pitch)
        return GeoInfo(self)
    
    def _init_additional_info(self, fov, heading, pitch):
        self.fov        = int(fov)
        self.heading    = int(heading)
        self.pitch      = int(pitch) # TODO: May not be necessary
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return self.get_name()
    
    def get_name(self) -> str:
        return f"{self.lat},{self.lng}-{self.fov}_{self.heading}_{self.pitch}" \
            if (self.fov>-1 or self.heading>-1 or self.pitch>-1) else self.get_latlng()
            
    def get_latlng(self) -> str:
        return f"{self.lat},{self.lng}"
    
    @classmethod
    def parse(cls, s:str):
        if s.count(".") == 3:
            s = os.path.splitext(s)[0]
        
        s = s.replace("-", "_").split("_")
        
        if len(s) < 4:
            return None
        try:
            lat,lng = s[0].split(",")
            fov     = s[1]
            heading = s[2]
            pitch   = s[3]
        except:
           lat,lng = -1, -1
           fov     = -1
           heading = -1
           pitch   = -1
        return GeoInfo((lat, lng), fov, heading, pitch)

def get_metadata(location:GeoInfo):
    baseURL = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": str(location), # 텍스트 문자열 or 위도, 경도 값
        'key': API_KEY
    }
    res = requests.get(baseURL, params=params)
    metadata = res.json()
    try:
        metadata['lat'] = metadata['location']['lat']
        metadata['lng'] = metadata['location']['lng']
        metadata['location'] = f"{metadata['lat']},{metadata['location']['lng']}"
    except:
        # print("Skipping")
        pass
    return metadata

def get_latlong_by_radius(location, radius_in_meter, rad_res):
    """Get latitude, longtitude tuples given location and it's radius in meter.

    Args:
        rad_res (int): Resolution of the radius.
        ang_res (float): Resolution of the angle.
    """
    # org_lat, org_lng = (float(n) for n in location.strip().split(","))
    org_lat, org_lng = location.lat, location.lng 
    """
    각각 입력된 원래 위도 경도 나타내는 듯?
    """
    # lat, long min, max
    lat_mm = \
        haversine.inverse_haversine((org_lat, org_lng), radius_in_meter, haversine.Direction.NORTH, unit=haversine.Unit.METERS)[0], \
        haversine.inverse_haversine((org_lat, org_lng), radius_in_meter, haversine.Direction.SOUTH, unit=haversine.Unit.METERS)[0]
    lng_mm = \
        haversine.inverse_haversine((org_lat, org_lng), radius_in_meter, haversine.Direction.WEST, unit=haversine.Unit.METERS)[1], \
        haversine.inverse_haversine((org_lat, org_lng), radius_in_meter, haversine.Direction.EAST, unit=haversine.Unit.METERS)[1]
    
    lat_mm, lng_mm = sorted(lat_mm), sorted(lng_mm)
    lat_res = (lat_mm[1]-lat_mm[0]) * rad_res / radius_in_meter
    lng_res = (lng_mm[1]-lng_mm[0]) * rad_res / radius_in_meter
    return lat_mm, lat_res, lng_mm, lng_res

def get_available_image_locations(location, radius, copyright="Google", rad_res = 8) -> List[GeoInfo]:
    
    lat_mm, lat_res, lng_mm, lng_res = get_latlong_by_radius(location, radius_in_meter=radius, rad_res=rad_res)
    # 주어진 Loacation에서 움직일 수 있는 범위와 resolution을 설정한다.
    print(lat_mm, lat_res, lng_mm, lng_res)
    metadatas = []
    grids = [f"{lat},{lng}" for lat,lng in tqdm(product(np.arange(lat_mm[0],lat_mm[1],lat_res), np.arange(lng_mm[0],lng_mm[1],lng_res)))]
    # min, max, res로 lat, lng 돌면서 검색 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        res = list(tqdm(exc.map(get_metadata, grids)))
    
    metadatas = [r for r in res if 'location' in r]
                
    new_sv_metadata_df = pd.DataFrame(metadatas)
    new_sv_metadata_df = new_sv_metadata_df.drop_duplicates(subset=['location'])

    if copyright is not None:
        new_sv_metadata_df = new_sv_metadata_df[new_sv_metadata_df['copyright'].str.contains(copyright)]

    available_locations = [GeoInfo(l) for l in new_sv_metadata_df['location']]
    # print("get_available_image_locations", available_locations, len(available_locations), i)
    # 주어진 위치에서 streetview를 얻어올 수 있는 위치를 알아내서 return
    print(f'available_locations: {available_locations}')
    return available_locations

def get_image(location, size='3280x2640', radius=50):
    baseURL = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": location.get_latlng(), # 텍스트 문자열 or 위도, 경도 값
        'size': size, # {widht}x{height}, 보니까 max 보다 크면 max 주는듯
        'heading': location.heading, #은 카메라의 나침반 방향을 나타냅니다. 허용되는 값은 0부터 360까지입니다. 두 값 모두 북쪽을 나타내고, 90은 동쪽, 180는 남쪽을 나타냅니다. If no heading is specified, 가장 가까운 사진이 찍힌 지점부터 지정된 location 방향으로 카메라를 전달하는 값이 계산됩니다.
        'fov': location.fov, #(기본값 90)은 이미지의 가로 시야를 결정합니다. 시야는 도 단위로 표현되며 허용되는 최댓값은 120입니다. 고정된 크기의 표시 영역을 처리하는 경우, 설정된 크기의 스트리트 뷰 이미지와 마찬가지로 시야는 확대/축소를 나타내며 숫자가 작을수록 확대/축소 수준이 더 높습니다.
        'pitch': location.pitch, # (기본값 0)는 스트리트 뷰 차량을 기준으로 카메라의 위 또는 아래 각도를 지정합니다. 이 값은 항상 그렇지는 않지만 대개는 완전 수평입니다. 양수 값은 카메라를 위로 올리고 (90도는 수직 위쪽을 나타냄), 음수 값은 카메라를 아래로 내립니다 (-90는 수직 아래쪽을 나타냄).
        'radius': radius, # (기본값 50)은 지정된 위도와 경도를 중심으로 파노라마를 검색할 반경으로 미터 단위로 지정됩니다. 유효한 값은 음이 아닌 정수입니다.
        'key': API_KEY
    }
    
    res = requests.get(baseURL, params=params, stream=True)
    if res.status_code == 200:
        image = np.asarray(bytearray(res.raw.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        return image
    else:
        print("Error!", res.text)


def get_image_and_save(location, size='3280x2640', radius=50, save_dir="./", use_cache=True):
    # warning! metadata 읽어서 나온 완전한 위,경도 값 아니면 caching이 제대로 안될 가능성 높음.
    
    if save_dir is not None:
        img_name = f"{save_dir}/{location}.png"
    
    image = get_image(location, size, radius)
    if save_dir is not None:
        img_name = f"{save_dir}/{location}.png"
        cv2.imwrite(img_name, image)
    return image

def get_360_image(location, fov, angle:float, heading_per=90, radius=50, save_dir="./", build_pano=True, use_cache=True):
    if 360%fov != 0:
        print(f"Warning! 360 is not divisible by {fov}")
    
    # for heading_bias in range(0, 120):
    heading_bias = 0
    # headings = list(range(0, 360, heading_per if heading_per else fov))
    headings = [angle+heading_per*idx for idx in range(4)]
    # 찾은 각도 기반으로 90도씩 회전하면서 이미지 추출
    pano_img = [None for _ in headings]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        fut_to_idxs = {exc.submit(get_image_and_save, location.update_additional_info(fov, heading+heading_bias, 0),save_dir=save_dir, use_cache=use_cache):i for i,heading in enumerate(headings)}
        for fut in as_completed(fut_to_idxs):
            i = fut_to_idxs[fut]
            img = fut.result()
            if build_pano:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                pano_img[i] = img
    if build_pano:
        os.makedirs(save_dir+"/pano", exist_ok=True)
        pano_img = cv2.hconcat(pano_img)
        cv2.imwrite(f'{save_dir}/pano/{location}_{radius}-{fov}-{heading_bias}-pano.png', pano_img)
        return pano_img



if __name__ == "__main__":
    # location= "37.3887, 126.6428" # lat/lng
    # location = "1.296056, 103.850541" # SMU 
    location = "37.4984, 127.0279" # Gangnam
    # location = "37.49830359656146,127.0280343987743"
    # location = "37.380489, 126.667467" # YONSEI
    # location = "37.390540, 126.644650"
    # location = "37.395093, 126.638768"

    latitude, longitude = location.split(',')
    if os.path.exists(f'imgs/{latitude}_{longitude}.png'):
        img = cv2.imread(f'imgs/{latitude}_{longitude}.png')
    else:
        data = get_map_image(latitude, longitude)
        img = bytes2img(data)
        cv2.imwrite(f'imgs/{latitude}_{longitude}.png', img)

    edges = cv2.Canny(img, 100, 200)
    point = find_closest_nonzero_point(edges)

    points = find_nonzero_in_window(edges, point, kernel_size=20)
    slope, intercept = find_line_from_points(points)
    angle = np.rad2deg(np.arctan(slope))[0]
    print(f"The angle of road is {angle}°")
    
    tmp = visualize_line(edges, slope, intercept)
    # cv2.imshow('fitted line', tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    save_dir = "./save_imgs"
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    sv_metadata_path = None

    fov      = 120

    available_img_locs = get_available_image_locations(GeoInfo(location), radius=6, rad_res=3)
    for location in available_img_locs[:1]:
        pano_img = get_360_image(location, fov, angle=angle, save_dir=save_dir, build_pano=False)
    
    