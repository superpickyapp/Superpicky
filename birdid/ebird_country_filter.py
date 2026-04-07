#!/usr/bin/env python3
"""
eBird国家鸟类过滤器

DEPRECATED: 此模块已被 avonet_filter.py 替代
保留代码仅供参考，不再被调用
"""
import warnings
warnings.warn(
    "ebird_country_filter 已弃用，请使用 avonet_filter",
    DeprecationWarning,
    stacklevel=2
)

import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import time

class eBirdCountryFilter:
    def __init__(self, api_key: str, cache_dir: str = "ebird_cache", offline_dir: str = "offline_ebird_data"):
        """
        初始化eBird国家过滤器

        Args:
            api_key: eBird API密钥
            cache_dir: 缓存目录路径
            offline_dir: 离线数据目录路径
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.offline_dir = offline_dir
        self.base_url = "https://api.ebird.org/v2"
        self.headers = {
            'x-ebirdapitoken': api_key
        }

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 确保离线数据目录存在
        os.makedirs(offline_dir, exist_ok=True)

        # 缓存有效期（天数）
        self.cache_validity_days = 30
        
        # 国家代码映射（ISO 2字母代码）
        self.country_codes = {
            'australia': 'AU',
            'china': 'CN', 
            'usa': 'US',
            'canada': 'CA',
            'brazil': 'BR',
            'india': 'IN',
            'indonesia': 'ID',
            'mexico': 'MX',
            'colombia': 'CO',
            'peru': 'PE',
            'ecuador': 'EC',
            'bolivia': 'BO',
            'venezuela': 'VE',
            'chile': 'CL',
            'argentina': 'AR',
            'south_africa': 'ZA',
            'kenya': 'KE',
            'tanzania': 'TZ',
            'madagascar': 'MG',
            'cameroon': 'CM',
            'ghana': 'GH',
            'nigeria': 'NG',
            'ethiopia': 'ET',
            'uganda': 'UG',
            'costa_rica': 'CR',
            'panama': 'PA',
            'guatemala': 'GT',
            'nicaragua': 'NI',
            'honduras': 'HN',
            'belize': 'BZ',
            'el_salvador': 'SV',
            'norway': 'NO',
            'sweden': 'SE',
            'finland': 'FI',
            'united_kingdom': 'GB',
            'france': 'FR',
            'spain': 'ES',
            'italy': 'IT',
            'germany': 'DE',
            'poland': 'PL',
            'romania': 'RO',
            'turkey': 'TR',
            'russia': 'RU',
            'japan': 'JP',
            'south_korea': 'KR',
            'thailand': 'TH',
            'vietnam': 'VN',
            'philippines': 'PH',
            'malaysia': 'MY',
            'singapore': 'SG',
            'new_zealand': 'NZ'
        }
    
    def get_cache_file_path(self, country_code: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"species_list_{country_code}.json")
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存时间
            cache_time = datetime.fromisoformat(cache_data.get('cached_at', '1970-01-01'))
            expiry_time = cache_time + timedelta(days=self.cache_validity_days)
            
            return datetime.now() < expiry_time
        except:
            return False
    
    def load_cached_species_list(self, country_code: str) -> Optional[Set[str]]:
        """从缓存加载物种列表"""
        cache_file = self.get_cache_file_path(country_code)

        if not self.is_cache_valid(cache_file):
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            species_set = set(cache_data.get('species', []))
            print(f"从缓存加载 {country_code} 的物种列表: {len(species_set)} 个物种")
            return species_set
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return None

    def load_offline_species_list(self, country_code: str) -> Optional[Set[str]]:
        """从离线数据加载物种列表"""
        offline_file = os.path.join(self.offline_dir, f"species_list_{country_code}.json")

        if not os.path.exists(offline_file):
            return None

        try:
            with open(offline_file, 'r', encoding='utf-8') as f:
                offline_data = json.load(f)

            species_set = set(offline_data.get('species', []))
            print(f"从离线数据加载 {country_code} 的物种列表: {len(species_set)} 个物种 (离线)")
            return species_set
        except Exception as e:
            print(f"加载离线数据失败: {e}")
            return None

    def is_offline_data_available(self) -> bool:
        """检查是否有离线数据可用"""
        index_file = os.path.join(self.offline_dir, "offline_index.json")
        return os.path.exists(index_file)

    def get_available_offline_countries(self) -> List[str]:
        """获取可用的离线国家列表"""
        if not self.is_offline_data_available():
            return []

        try:
            index_file = os.path.join(self.offline_dir, "offline_index.json")
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            return list(index_data.get('countries', {}).keys())
        except Exception as e:
            print(f"读取离线数据索引失败: {e}")
            return []
    
    def save_species_list_to_cache(self, country_code: str, species_list: List[str]) -> None:
        """保存物种列表到缓存"""
        cache_file = self.get_cache_file_path(country_code)
        
        cache_data = {
            'country_code': country_code,
            'species': list(species_list),
            'species_count': len(species_list),
            'cached_at': datetime.now().isoformat(),
            'api_version': '2.0'
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"已缓存 {country_code} 的物种列表: {len(species_list)} 个物种")
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def fetch_species_list_from_api(self, country_code: str) -> Optional[List[str]]:
        """从eBird API获取物种列表"""
        url = f"{self.base_url}/product/spplist/{country_code}"

        try:
            print(f"正在从eBird API获取 {country_code} 的物种列表...")
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                species_list = response.json()
                print(f"成功获取 {country_code} 的 {len(species_list)} 个物种")

                # 添加短暂延迟以避免API限制
                time.sleep(0.1)

                return species_list
            elif response.status_code == 404:
                print(f"国家代码 {country_code} 未找到")
                return None
            elif response.status_code == 429:
                print("API请求限制，请稍后再试")
                return None
            else:
                print(f"API请求失败: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            return None

    def get_region_code_from_gps(self, lat: float, lon: float) -> tuple:
        """
        根据GPS坐标获取eBird区域代码和国家代码

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            (region_code, country_code) 元组
            - region_code: 如 "AU-NT" (澳大利亚北领地)
            - country_code: 如 "AU" (澳大利亚)
            失败返回 (None, None)
        """
        # 首先尝试离线判定（基于经纬度边界，无需网络）
        region_code, country_name = self._offline_region_detection(lat, lon)
        if region_code:
            print(f"[GPS] 离线检测区域: {region_code} ({country_name})")
            country_code = region_code.split('-')[0] if '-' in region_code else region_code
            return region_code, country_code
        
        # 回退到 Nominatim API（需要网络）
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'accept-language': 'en'
            }
            headers = {
                'User-Agent': 'SuperBirdID/1.0'
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                country_code = address.get('country_code', '').upper()

                if not country_code:
                    print(f"无法从坐标 ({lat}, {lon}) 获取国家代码")
                    return None, None

                state = (address.get('state') or
                        address.get('province') or
                        address.get('region') or
                        address.get('territory') or
                        address.get('state_district'))

                if state:
                    state_code = self.map_state_to_ebird_code(country_code, state)
                    if state_code:
                        region_code = f"{country_code}-{state_code}"
                        print(f"[GPS] Nominatim检测: {country_code} / {state} → {region_code}")
                        return region_code, country_code
                    else:
                        print(f"[GPS] Nominatim检测: {country_code} / {state} (无法映射州/省代码)")
                        return country_code, country_code
                else:
                    print(f"[GPS] Nominatim检测: {country_code} (无州/省信息)")
                    return country_code, country_code

        except Exception as e:
            print(f"[GPS] Nominatim API失败: {e}")

        return None, None
    
    def _offline_region_detection(self, lat: float, lon: float) -> tuple:
        """
        离线判定区域（基于经纬度边界，无需网络）
        支持澳大利亚、美国、中国等常见国家的州/省判定
        """
        # 澳大利亚各州边界（大致范围）
        au_states = {
            'AU-QLD': {'name': 'Queensland', 'lat': (-29, -10), 'lon': (138, 154)},
            'AU-NSW': {'name': 'New South Wales', 'lat': (-37.5, -28), 'lon': (141, 154)},
            'AU-VIC': {'name': 'Victoria', 'lat': (-39.2, -34), 'lon': (141, 150)},
            'AU-TAS': {'name': 'Tasmania', 'lat': (-43.7, -39.5), 'lon': (143.5, 148.5)},
            'AU-SA': {'name': 'South Australia', 'lat': (-38, -26), 'lon': (129, 141)},
            'AU-WA': {'name': 'Western Australia', 'lat': (-35, -13.5), 'lon': (112.5, 129)},
            'AU-NT': {'name': 'Northern Territory', 'lat': (-26, -10.5), 'lon': (129, 138)},
            'AU-ACT': {'name': 'Australian Capital Territory', 'lat': (-35.95, -35.1), 'lon': (148.75, 149.4)},
        }
        
        # 检查澳大利亚
        if -44 <= lat <= -10 and 112 <= lon <= 155:
            for region_code, info in au_states.items():
                lat_min, lat_max = info['lat']
                lon_min, lon_max = info['lon']
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    return region_code, info['name']
            # 在澳大利亚但无法确定具体州
            return 'AU', 'Australia'
        
        # 新西兰
        if -47.5 <= lat <= -34 and 166 <= lon <= 179:
            return 'NZ', 'New Zealand'
        
        # 中国
        if 18 <= lat <= 54 and 73 <= lon <= 135:
            return 'CN', 'China'
        
        # 日本
        if 24 <= lat <= 46 and 122 <= lon <= 154:
            return 'JP', 'Japan'
        
        # 美国本土（大致范围）
        if 24 <= lat <= 49 and -125 <= lon <= -66:
            return 'US', 'United States'
        
        # 英国
        if 49 <= lat <= 61 and -8 <= lon <= 2:
            return 'GB', 'United Kingdom'
        
        return None, None

    def map_state_to_ebird_code(self, country_code: str, state_name: str) -> Optional[str]:
        """
        将州/省名称映射到eBird代码

        Args:
            country_code: 国家代码（如 "AU"）
            state_name: 州/省名称（如 "Northern Territory"）

        Returns:
            eBird州/省代码（如 "NT"），失败返回None
        """
        # 常见国家的州/省映射
        state_mappings = {
            'AU': {  # 澳大利亚
                'Northern Territory': 'NT',
                'New South Wales': 'NSW',
                'Queensland': 'QLD',
                'South Australia': 'SA',
                'Tasmania': 'TAS',
                'Victoria': 'VIC',
                'Western Australia': 'WA',
                'Australian Capital Territory': 'ACT',
            },
            'US': {  # 美国
                'California': 'CA',
                'New York': 'NY',
                'Texas': 'TX',
                'Florida': 'FL',
                # ... 可以继续添加更多州
            },
            'CN': {  # 中国（使用数字代码）
                'Beijing': '11',
                'Shanghai': '31',
                'Guangdong': '44',
                'Sichuan': '51',
                # ... 可以继续添加更多省份
            },
            'CA': {  # 加拿大
                'Ontario': 'ON',
                'Quebec': 'QC',
                'British Columbia': 'BC',
                'Alberta': 'AB',
                # ... 可以继续添加更多省份
            }
        }

        if country_code not in state_mappings:
            return None

        # 尝试精确匹配
        state_map = state_mappings[country_code]
        if state_name in state_map:
            return state_map[state_name]

        # 尝试部分匹配（忽略大小写）
        state_lower = state_name.lower()
        for full_name, code in state_map.items():
            if full_name.lower() in state_lower or state_lower in full_name.lower():
                return code

        return None

    def get_location_cache_file_path(self, lat: float, lon: float, radius: int) -> str:
        """获取位置缓存文件路径"""
        # 使用坐标和半径创建唯一的缓存文件名
        lat_str = f"{lat:.3f}".replace('.', '_').replace('-', 'n')
        lon_str = f"{lon:.3f}".replace('.', '_').replace('-', 'n')
        return os.path.join(self.cache_dir, f"location_{lat_str}_{lon_str}_{radius}km.json")

    def fetch_species_list_by_location(self, lat: float, lon: float, radius: int = 25) -> Optional[dict]:
        """
        基于GPS坐标从eBird API获取附近区域的物种列表

        Args:
            lat: 纬度
            lon: 经度
            radius: 搜索半径（公里，最大50km）

        Returns:
            字典 {'species': List[str], 'observation_count': int} 或 None
        """
        # 限制半径范围
        radius = min(max(radius, 1), 50)

        # 使用eBird的近期观察记录API
        # 注意：eBird API的back参数支持1-30天
        # 由于没有直接的全年GPS位置API，我们使用nearby hotspots + region species list的组合策略
        url = f"{self.base_url}/data/obs/geo/recent"

        params = {
            'lat': lat,
            'lng': lon,
            'dist': radius,
            'back': 30,  # 获取最近30天的观察记录
            'maxResults': 10000,  # 获取更多结果
            'includeProvisional': 'false',  # 只包含审核通过的记录
            'fmt': 'json'
        }

        try:
            print(f"正在获取位置 ({lat:.3f}, {lon:.3f}) 半径 {radius}km 内最近30天的鸟类观察记录...")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 200:
                observations = response.json()
                species_codes = list(set(obs['speciesCode'] for obs in observations))

                print(f"✓ 最近30天: {len(species_codes)} 个物种（{len(observations)} 条观察记录）")

                return {
                    'species': species_codes,
                    'observation_count': len(observations)
                }
            elif response.status_code == 404:
                print(f"位置 ({lat}, {lon}) 未找到观察记录")
                return None
            elif response.status_code == 429:
                print("⚠️ API请求限制")
                return None
            else:
                print(f"⚠️ API请求失败: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            return None

    def get_location_species_list(self, lat: float, lon: float, radius: int = 25) -> Optional[Set[str]]:
        """
        获取GPS位置附近的鸟类物种列表（三级回退策略）

        策略优先级：
        1. GPS位置30天数据（最精确，但物种少）
        2. GPS所在区域的年度数据（较精确，物种较多）
        3. GPS所在国家的离线数据（兜底方案，物种最多）

        Args:
            lat: 纬度
            lon: 经度
            radius: 搜索半径（公里）

        Returns:
            物种代码集合，如果获取失败返回None
        """
        # 检查缓存
        cache_file = self.get_location_cache_file_path(lat, lon, radius)

        if self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                species_set = set(cache_data.get('species', []))
                data_source = cache_data.get('data_source', 'unknown')
                print(f"从缓存加载位置物种列表: {len(species_set)} 个物种 (来源: {data_source})")
                return species_set
            except Exception as e:
                print(f"加载位置缓存失败: {e}")

        # 缓存无效或不存在，使用三级回退策略
        species_list = None
        data_source = None

        # 第1级：尝试获取GPS位置30天数据
        print("策略1: 获取GPS位置30天数据...")
        api_result = self.fetch_species_list_by_location(lat, lon, radius)

        # 定义一个合理的阈值：如果30天数据少于50个物种，认为数据不足
        MIN_SPECIES_THRESHOLD = 50

        if api_result and 'species' in api_result and len(api_result['species']) >= MIN_SPECIES_THRESHOLD:
            species_list = api_result['species']
            data_source = "GPS位置30天数据"
            observation_count = api_result.get('observation_count', len(species_list))
            print(f"✓ 使用GPS位置30天数据: {len(species_list)} 个物种")
        else:
            # GPS 30天数据为空或不足，尝试区域年度数据
            gps_species_count = len(api_result.get('species', [])) if api_result else 0
            if gps_species_count > 0:
                print(f"⚠️ GPS位置30天数据仅 {gps_species_count} 个物种，尝试区域年度数据...")
            else:
                print("⚠️ GPS位置30天数据为空，尝试区域年度数据...")

            # 第2级：尝试获取GPS所在区域的年度数据
            region_code, country_code = self.get_region_code_from_gps(lat, lon)

            if region_code:
                print(f"策略2: 获取区域 {region_code} 的年度数据...")
                region_species = self.get_country_species_list(region_code)

                if region_species and len(region_species) > 0:
                    species_list = list(region_species)
                    data_source = f"区域{region_code}年度数据"
                    observation_count = len(species_list)
                    print(f"✓ 使用区域年度数据: {len(species_list)} 个物种")
                else:
                    print(f"⚠️ 区域 {region_code} 年度数据获取失败")

            # 第3级：如果区域数据也失败，使用国家离线数据
            if not species_list and country_code:
                print("策略3: 使用国家离线数据作为兜底...")
                country_species = self.load_offline_species_list(country_code)
                if country_species and len(country_species) > 0:
                    species_list = list(country_species)
                    data_source = f"国家{country_code}离线数据"
                    observation_count = len(species_list)
                    print(f"✓ 使用国家离线数据: {len(species_list)} 个物种")
                else:
                    print(f"⚠️ 国家 {country_code} 离线数据不可用")

        # 保存到缓存
        if species_list:
            cache_data = {
                'lat': lat,
                'lon': lon,
                'radius': radius,
                'species': species_list,
                'species_count': len(species_list),
                'observation_count': observation_count if 'observation_count' in locals() else len(species_list),
                'data_source': data_source,
                'cached_at': datetime.now().isoformat(),
                'api_version': '2.0'
            }

            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                print(f"已缓存位置物种列表: {len(species_list)} 个物种 (来源: {data_source})")
            except Exception as e:
                print(f"保存位置缓存失败: {e}")

            return set(species_list)

        print("❌ 所有策略均失败，无法获取物种数据")
        return None

    def get_location_cache_info(self, lat: float, lon: float, radius: int = 25) -> Optional[dict]:
        """
        获取位置缓存的详细信息

        Returns:
            字典包含 {'species_count': int, 'observation_count': int, 'cached_at': str, 'data_source': str} 或 None
        """
        cache_file = self.get_location_cache_file_path(lat, lon, radius)

        if self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                return {
                    'species_count': cache_data.get('species_count', len(cache_data.get('species', []))),
                    'observation_count': cache_data.get('observation_count', cache_data.get('species_count', 0)),
                    'cached_at': cache_data.get('cached_at', ''),
                    'data_source': cache_data.get('data_source', 'unknown')  # 添加data_source字段
                }
            except:
                return None
        return None

    def get_country_species_list(self, country_input: str) -> Optional[Set[str]]:
        """
        获取国家或区域的鸟类物种列表（优先级：缓存 > API > 离线数据）

        Args:
            country_input: 国家名称、国家代码或区域代码（如 "AU", "AU-NT"）

        Returns:
            物种代码集合，如果获取失败返回None
        """
        # 标准化输入
        country_input = country_input.strip()

        # 检查是否是区域代码格式（如 "AU-NT", "US-CA"）
        if '-' in country_input:
            # 这是区域代码，直接使用（转大写）
            region_code = country_input.upper()
            print(f"检测到区域代码: {region_code}")

            # 1. 首先尝试从缓存加载
            cached_species = self.load_cached_species_list(region_code)
            if cached_species is not None:
                return cached_species

            # 2. 缓存无效，尝试从API获取
            species_list = self.fetch_species_list_from_api(region_code)

            if species_list:
                # 保存到缓存
                self.save_species_list_to_cache(region_code, species_list)
                return set(species_list)

            # 3. API失败
            print(f"❌ 无法获取 {region_code} 的鸟类数据")
            return None

        # 不是区域代码，按国家代码处理
        country_input_lower = country_input.lower()

        # 尝试直接作为国家代码使用
        country_code = country_input.upper()

        # 如果不是2位代码，尝试从映射中查找
        if len(country_code) != 2 or not country_code.isalpha():
            country_code = self.country_codes.get(country_input_lower)

            if not country_code:
                # 尝试部分匹配
                for key, code in self.country_codes.items():
                    if country_input_lower in key or key in country_input_lower:
                        country_code = code
                        break

        if not country_code:
            print(f"未找到国家代码: {country_input}")
            print("支持的国家:")
            for name, code in list(self.country_codes.items())[:10]:
                print(f"  {name}: {code}")
            print("  ... 等更多国家")
            return None

        # 1. 首先尝试从缓存加载
        cached_species = self.load_cached_species_list(country_code)
        if cached_species is not None:
            return cached_species

        # 2. 缓存无效，尝试从API获取
        species_list = self.fetch_species_list_from_api(country_code)

        if species_list:
            # 保存到缓存
            self.save_species_list_to_cache(country_code, species_list)
            return set(species_list)

        # 3. API失败，尝试从离线数据加载
        print(f"API获取失败，尝试使用离线数据...")
        offline_species = self.load_offline_species_list(country_code)
        if offline_species is not None:
            return offline_species

        print(f"❌ 无法获取 {country_code} 的鸟类数据（尝试了缓存、API和离线数据）")
        return None
    
    def filter_results_by_country(self, recognition_results: List[Dict], country_species: Set[str]) -> List[Dict]:
        """
        使用国家物种列表过滤识别结果
        
        Args:
            recognition_results: 鸟类识别结果列表
            country_species: 国家物种代码集合
            
        Returns:
            过滤后的结果列表
        """
        filtered_results = []
        
        for result in recognition_results:
            # 假设结果中包含物种的eBird代码
            # 这需要根据实际的识别结果格式调整
            species_code = result.get('ebird_code') or result.get('species_code')
            
            if not species_code:
                # 如果没有eBird代码，尝试从英文名称推断
                english_name = result.get('english_name', '')
                if english_name:
                    # 这里需要实现英文名称到eBird代码的映射
                    # 或者使用其他方法匹配
                    pass
            
            # 检查物种是否在国家列表中
            if species_code and species_code in country_species:
                result['country_match'] = True
                result['confidence_boost'] = 1.2  # 给予20%的置信度提升
                filtered_results.append(result)
            elif not species_code:
                # 如果无法确定物种代码，保留但不提升
                result['country_match'] = False
                filtered_results.append(result)
        
        # 按照是否匹配国家和置信度排序
        filtered_results.sort(
            key=lambda x: (x.get('country_match', False), x.get('confidence', 0)), 
            reverse=True
        )
        
        return filtered_results
    
    def get_all_countries(self) -> Optional[List[Dict[str, str]]]:
        """
        从eBird API获取全球所有国家/地区列表

        Returns:
            国家列表 [{'code': 'AU', 'name': 'Australia'}, ...]
        """
        url = f"{self.base_url}/ref/region/list/country/world"

        try:
            print("正在获取全球国家列表...")
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                countries = response.json()
                print(f"✓ 获取成功: {len(countries)} 个国家/地区")
                return countries
            else:
                print(f"⚠️ 获取失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ 请求失败: {e}")
            return None

    def get_subnational_regions(self, country_code: str) -> Optional[List[Dict[str, str]]]:
        """
        获取指定国家的二级区域列表（州/省）

        Args:
            country_code: 国家代码（如 'AU', 'CN', 'US'）

        Returns:
            区域列表 [{'code': 'AU-NSW', 'name': 'New South Wales'}, ...]
        """
        url = f"{self.base_url}/ref/region/list/subnational1/{country_code}"

        try:
            print(f"正在获取 {country_code} 的二级区域列表...")
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                regions = response.json()
                print(f"✓ 获取成功: {len(regions)} 个区域")
                return regions
            elif response.status_code == 404:
                print(f"⚠️ {country_code} 没有二级区域数据")
                return []
            else:
                print(f"⚠️ 获取失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ 请求失败: {e}")
            return None

    def get_supported_countries(self) -> Dict[str, str]:
        """获取支持的国家列表（已弃用，建议使用 get_all_countries）"""
        return self.country_codes.copy()
    
    def clear_cache(self, country_code: str = None) -> None:
        """清理缓存"""
        if country_code:
            cache_file = self.get_cache_file_path(country_code)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"已清理 {country_code} 的缓存")
        else:
            # 清理所有缓存
            for file in os.listdir(self.cache_dir):
                if file.startswith('species_list_') and file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            print("已清理所有缓存")


def test_ebird_filter():
    """测试eBird过滤器"""
    import os
    # eBird API密钥（优先使用环境变量，否则使用默认值）
    api_key = os.environ.get('EBIRD_API_KEY', '60nan25sogpo')
    filter_system = eBirdCountryFilter(api_key)
    
    # 测试获取澳洲鸟类列表
    print("=== 测试澳洲鸟类列表 ===")
    au_species = filter_system.get_country_species_list("australia")
    if au_species:
        print(f"澳洲鸟类物种数量: {len(au_species)}")
        print("前10个物种代码:", list(au_species)[:10])
    
    # 测试获取中国鸟类列表
    print("\n=== 测试中国鸟类列表 ===")
    cn_species = filter_system.get_country_species_list("china")
    if cn_species:
        print(f"中国鸟类物种数量: {len(cn_species)}")
        print("前10个物种代码:", list(cn_species)[:10])


if __name__ == "__main__":
    test_ebird_filter()