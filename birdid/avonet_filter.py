#!/usr/bin/env python3
"""
AvonetFilter - 基于 avonet.db 的离线物种过滤器

使用 AVONET 全球鸟类分布数据进行离线物种过滤，
替代需要网络连接的 eBird API。

数据库结构：
- distributions: 物种-网格映射 (species, worldid)
- places: 1x1 度网格边界 (worldid, south, north, west, east)
- sp_cls_map: 物种名 -> OSEA class_id 映射 (species, cls)
"""

import os
import sqlite3
from typing import Set, List, Optional, Tuple
from tools.i18n import t as _t

# 区域边界定义 (south, north, west, east)
# 格式: REGION_CODE: (南纬界, 北纬界, 西经界, 东经界)
REGION_BOUNDS = {
    # 全球
    "GLOBAL": (-90, 90, -180, 180),

    # 六大洲 (宽泛定义，用于大范围检索)
    "AF": (-35, 37, -17, 51),        # 非洲 (Africa)
    "AS": (-10, 81, 26, 170),        # 亚洲 (Asia)
    "EU": (34, 71, -25, 45),         # 欧洲 (Europe)
    "NA": (14, 83, -168, -52),       # 北美洲 (North America)
    "SA": (-56, 13, -81, -34),       # 南美洲 (South America)
    "OC": (-47, -10, 110, 180),      # 大洋洲 (Oceania)

    # 亚太地区 - 国家
    "AU": (-44, -10, 112, 155),      # 澳大利亚
    "NZ": (-47.5, -34, 166, 179),    # 新西兰
    "CN": (18, 54, 73, 135),         # 中国
    "JP": (24, 46, 122, 154),        # 日本
    "KR": (33, 43, 124, 132),        # 韩国
    "TW": (21.5, 25.5, 119, 122.5),  # 台湾
    "HK": (22.1, 22.6, 113.8, 114.5), # 香港
    "TH": (5.5, 20.5, 97.5, 105.5),  # 泰国
    "MY": (0.5, 7.5, 99.5, 119.5),   # 马来西亚
    "SG": (1.1, 1.5, 103.6, 104.1),  # 新加坡
    "ID": (-11, 6, 95, 141),         # 印度尼西亚
    "PH": (4.5, 21, 116, 127),       # 菲律宾
    "VN": (8, 23.5, 102, 110),       # 越南
    "IN": (6, 36, 68, 98),           # 印度
    "LK": (5, 10, 79, 82),           # 斯里兰卡
    "NP": (26, 31, 80, 88),          # 尼泊尔
    "MN": (41, 52, 87, 120),         # 蒙古
    "RU": (41, 82, 19, 180),         # 俄罗斯

    # 美洲
    "US": (24, 49, -125, -66),       # 美国本土
    "CA": (42, 83, -141, -52),       # 加拿大
    "MX": (14, 33, -118, -86),       # 墨西哥
    "BR": (-34, 5.5, -74, -34),      # 巴西
    "AR": (-55, -21, -73, -53),      # 阿根廷
    "CL": (-56, -17, -76, -66),      # 智利
    "CO": (-4.5, 13, -79, -66),      # 哥伦比亚
    "PE": (-18.5, 0, -81, -68),      # 秘鲁
    "EC": (-5, 2, -81, -75),         # 厄瓜多尔
    "CR": (8, 11.5, -86, -82.5),     # 哥斯达黎加

    # 欧洲
    "GB": (49, 61, -8, 2),           # 英国
    "FR": (41, 51.5, -5, 10),        # 法国
    "DE": (47, 55.5, 5.5, 15.5),     # 德国
    "ES": (35.5, 44, -10, 4.5),      # 西班牙
    "IT": (36, 47.5, 6.5, 18.5),     # 意大利
    "NO": (57.5, 71.5, 4.5, 31.5),   # 挪威
    "SE": (55, 69.5, 10.5, 24.5),    # 瑞典
    "FI": (59.5, 70.5, 19.5, 31.5),  # 芬兰
    "PL": (49, 55, 14, 24.5),        # 波兰
    "TR": (35.5, 42.5, 25.5, 45),    # 土耳其
    "PT": (36, 42, -10, -6),         # 葡萄牙
    "NL": (50, 54, 3, 8),            # 荷兰
    "CH": (45, 48, 5, 11),           # 瑞士
    "GR": (34, 42, 19, 29),          # 希腊
    "UA": (44, 53, 22, 41),          # 乌克兰

    # 非洲
    "MG": (-26, -11, 43, 51),        # 马达加斯加
    "ZA": (-35, -22, 16.5, 33),      # 南非
    "KE": (-5, 5, 33.5, 42),         # 肯尼亚
    "TZ": (-12, -1, 29, 41),         # 坦桑尼亚
    "EG": (22, 32, 24.5, 37),        # 埃及
    "MA": (27, 36, -13, -1),         # 摩洛哥

    # 澳大利亚各州
    "AU-QLD": (-29,   -10,  138,    154),    # Queensland
    "AU-NSW": (-37.5, -28,  141,    154),    # New South Wales
    "AU-VIC": (-39.2, -34,  141,    150),    # Victoria
    "AU-TAS": (-43.7, -39.5, 143.5, 148.5), # Tasmania
    "AU-SA":  (-38,   -26,  129,    141),    # South Australia
    "AU-WA":  (-35,   -13.5, 112.5, 129),   # Western Australia
    "AU-NT":  (-26,   -10.5, 129,   138),   # Northern Territory
    "AU-ACT": (-35.95,-35.1, 148.75,149.4), # Australian Capital Territory

    # 美国各州 (south, north, west, east)
    "US-AL": (30,   35,   -88.5, -84.9), "US-AK": (51,   72,   -168,  -130),
    "US-AZ": (31.3, 37,   -114.8,-109),  "US-AR": (33,   36.5, -94.6, -89.6),
    "US-CA": (32.5, 42,   -124.5,-114),  "US-CO": (37,   41,   -109,  -102),
    "US-CT": (40.9, 42.1, -73.7, -71.8),"US-DE": (38.4, 39.8, -75.8, -75),
    "US-FL": (24.4, 31,   -87.7, -80),   "US-GA": (30.4, 35,   -85.6, -80.8),
    "US-HI": (18.9, 22.2, -160.3,-154.8),"US-ID": (42,   49,   -117.2,-111),
    "US-IL": (36.9, 42.5, -91.5, -87.5),"US-IN": (37.8, 41.8, -88.1, -84.8),
    "US-IA": (40.4, 43.5, -96.6, -90.1),"US-KS": (37,   40,   -102.1,-94.6),
    "US-KY": (36.5, 39.2, -89.6, -81.9),"US-LA": (28.9, 33.1, -94.1, -88.8),
    "US-ME": (43.1, 47.5, -71.1, -66.9),"US-MD": (37.9, 39.7, -79.5, -75),
    "US-MA": (41.2, 42.9, -73.5, -69.9),"US-MI": (41.7, 48.3, -90.4, -82.4),
    "US-MN": (43.5, 49.4, -97.2, -89.5),"US-MS": (30,   35,   -91.7, -88.1),
    "US-MO": (36,   40.6, -95.8, -89.1),"US-MT": (44.4, 49,   -116.1,-104),
    "US-NE": (40,   43,   -104.1,-95.3),"US-NV": (35,   42,   -120,  -114),
    "US-NH": (42.7, 45.3, -72.6, -70.7),"US-NJ": (38.9, 41.4, -75.6, -73.9),
    "US-NM": (31.3, 37,   -109.1,-103),  "US-NY": (40.5, 45.1, -79.8, -71.9),
    "US-NC": (33.8, 36.6, -84.3, -75.5),"US-ND": (45.9, 49,   -104.1,-96.6),
    "US-OH": (38.4, 42,   -84.8, -80.5),"US-OK": (33.6, 37,   -103,  -94.4),
    "US-OR": (41.9, 46.3, -124.6,-116.5),"US-PA": (39.7, 42.3, -80.5, -74.7),
    "US-RI": (41.1, 42.1, -71.9, -71.1),"US-SC": (32,   35.2, -83.4, -78.5),
    "US-SD": (42.5, 45.9, -104.1,-96.4),"US-TN": (35,   36.7, -90.3, -81.6),
    "US-TX": (25.8, 36.5, -106.6,-93.5),"US-UT": (37,   42,   -114.1,-109),
    "US-VT": (42.7, 45.1, -73.4, -71.5),"US-VA": (36.5, 39.5, -83.7, -75.2),
    "US-WA": (45.5, 49,   -124.8,-116.9),"US-WV": (37.2, 40.6, -82.7, -77.7),
    "US-WI": (42.5, 47.1, -92.9, -86.8),"US-WY": (41,   45,   -111.1,-104),

    # 中国各省（south, north, west, east）
    "CN-11": (39.4, 41.1, 115.4, 117.7),  # 北京
    "CN-12": (38.6, 40.3, 116.7, 118.1),  # 天津
    "CN-13": (36,   42.7, 113.5, 119.8),  # 河北
    "CN-14": (34.6, 40.7, 110.2, 114.6),  # 山西
    "CN-15": (37.5, 53.3, 97.2,  126.1),  # 内蒙古
    "CN-21": (38.7, 43.5, 118.8, 125.7),  # 辽宁
    "CN-22": (41.2, 46,   121.6, 131.3),  # 吉林
    "CN-23": (43.4, 53.6, 121.1, 135.1),  # 黑龙江
    "CN-31": (30.7, 31.9, 120.8, 122),    # 上海
    "CN-32": (30.8, 35.1, 116.4, 121.9),  # 江苏
    "CN-33": (27.1, 31.2, 118.1, 122.9),  # 浙江
    "CN-34": (29.4, 34.7, 114.9, 119.9),  # 安徽
    "CN-35": (23.5, 28.3, 115.8, 120.7),  # 福建
    "CN-36": (24.5, 30.1, 113.6, 118.5),  # 江西
    "CN-37": (34.4, 38.3, 114.8, 122.7),  # 山东
    "CN-41": (31.4, 36.4, 110.4, 116.7),  # 河南
    "CN-42": (29.1, 33.2, 108.4, 116.1),  # 湖北
    "CN-43": (24.6, 30.1, 108.8, 114.3),  # 湖南
    "CN-44": (20.2, 25.5, 109.7, 117.3),  # 广东
    "CN-45": (20.9, 26.4, 104.5, 112.1),  # 广西
    "CN-46": (18.1, 20.2, 108.4, 111.2),  # 海南
    "CN-50": (28.2, 32.2, 105.3, 110.2),  # 重庆
    "CN-51": (26,   34.3, 97.4,  108.5),  # 四川
    "CN-52": (24.6, 29.2, 103.6, 109.6),  # 贵州
    "CN-53": (21.1, 29.3, 97.5,  106.2),  # 云南
    "CN-54": (26.8, 36.5, 78.4,  99.1),   # 西藏
    "CN-61": (31.7, 39.6, 105.5, 111.3),  # 陕西
    "CN-62": (32.6, 42.8, 92.4,  108.7),  # 甘肃
    "CN-63": (31.6, 39.2, 89.4,  103.1),  # 青海
    "CN-64": (35.2, 39.4, 104.3, 107.7),  # 宁夏
    "CN-65": (34.3, 49.2, 73.5,  96.4),   # 新疆
}


class AvonetFilter:
    """
    基于 AVONET 数据库的离线物种过滤器

    使用 1x1 度网格的鸟类分布数据，支持：
    - GPS 坐标查询：返回该位置可能出现的物种
    - 区域代码查询：返回指定区域的物种列表
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 AvonetFilter

        Args:
            db_path: avonet.db 的路径，如果为 None 则自动定位
        """
        if db_path is None:
            # 自动定位数据库文件
            db_path = self._find_database()

        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ebird_cls_map: Optional[dict] = None  # eBird code -> class_id（懒加载）

        # 尝试连接数据库
        if self.db_path and os.path.exists(self.db_path):
            try:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            except sqlite3.Error as e:
                print(_t("logs.avonet_db_failed", e=e))
                self._conn = None

    def _find_database(self) -> Optional[str]:
        """
        自动查找 avonet.db 文件

        查找顺序：
        1. birdid/data/avonet.db (相对于当前文件)
        2. data/avonet.db (相对于当前工作目录)
        3. 常见安装位置
        """
        # 相对于当前模块的位置
        module_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(module_dir, "data", "avonet.db"),
            os.path.join(module_dir, "..", "data", "avonet.db"),
            os.path.join(os.getcwd(), "birdid", "data", "avonet.db"),
            os.path.join(os.getcwd(), "data", "avonet.db"),
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        return None

    def is_available(self) -> bool:
        """
        检查数据库是否可用

        Returns:
            True 如果数据库连接正常且包含数据
        """
        if self._conn is None:
            return False

        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM sp_cls_map")
            count = cursor.fetchone()[0]
            return count > 0
        except sqlite3.Error:
            return False

    def get_species_by_gps(self, lat: float, lon: float) -> Set[int]:
        """
        根据 GPS 坐标获取该位置可能出现的物种 class_ids

        使用 1x1 度网格查询，返回所有在该网格中有分布记录的物种。

        Args:
            lat: 纬度 (-90 到 90)
            lon: 经度 (-180 到 180)

        Returns:
            物种 class_id 的集合，如果查询失败返回空集合
        """
        if self._conn is None:
            return set()

        try:
            # 查询包含该GPS点的网格中的所有物种
            query = """
                SELECT DISTINCT sm.cls
                FROM distributions d
                JOIN places p ON d.worldid = p.worldid
                JOIN sp_cls_map sm ON d.species = sm.species
                WHERE ? BETWEEN p.south AND p.north
                  AND ? BETWEEN p.west AND p.east
            """
            cursor = self._conn.execute(query, (lat, lon))
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            print(_t("logs.avonet_gps_failed", e=e))
            return set()

    def get_species_by_region(self, region_code: str) -> Set[int]:
        """
        根据区域代码获取该区域的物种 class_ids

        Args:
            region_code: 区域代码 (如 "AU", "AU-SA", "CN", "JP")

        Returns:
            物种 class_id 的集合，如果区域不支持返回空集合
        """
        region_code = region_code.upper()

        if region_code not in REGION_BOUNDS:
            print(_t("logs.avonet_unsupported_region", code=region_code))
            return set()

        bounds = REGION_BOUNDS[region_code]
        return self._get_species_by_bounds(*bounds)

    def _get_species_by_bounds(
        self, south: float, north: float, west: float, east: float
    ) -> Set[int]:
        """
        根据边界框查询物种 class_ids

        查询所有与边界框有重叠的网格中的物种。

        Args:
            south: 南边界纬度
            north: 北边界纬度
            west: 西边界经度
            east: 东边界经度

        Returns:
            物种 class_id 的集合
        """
        if self._conn is None:
            return set()

        try:
            # 查询与边界框重叠的所有网格中的物种
            query = """
                SELECT DISTINCT sm.cls
                FROM distributions d
                JOIN places p ON d.worldid = p.worldid
                JOIN sp_cls_map sm ON d.species = sm.species
                WHERE p.north >= ? AND p.south <= ?
                  AND p.east >= ? AND p.west <= ?
            """
            cursor = self._conn.execute(query, (south, north, west, east))
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            print(_t("logs.avonet_bbox_failed", e=e))
            return set()

    def get_supported_regions(self) -> List[str]:
        """
        获取支持的区域代码列表

        Returns:
            支持的区域代码列表，按字母顺序排序
        """
        return sorted(REGION_BOUNDS.keys())

    def get_region_bounds(self, region_code: str) -> Optional[Tuple[float, float, float, float]]:
        """
        获取区域的边界坐标

        Args:
            region_code: 区域代码

        Returns:
            (south, north, west, east) 元组，如果区域不存在返回 None
        """
        return REGION_BOUNDS.get(region_code.upper())

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            finally:
                self._conn = None

    def __enter__(self):
        """支持 context manager 协议"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时关闭连接"""
        self.close()
        return False

    def __del__(self):
        """析构时关闭连接"""
        self.close()

    # ==================== eBird 国家级回退 ====================

    def _load_ebird_cls_map(self) -> dict:
        """懒加载 ebird_classid_mapping.json，返回 ebird_code -> class_id 的反向映射"""
        if self._ebird_cls_map is not None:
            return self._ebird_cls_map

        module_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(module_dir, "data", "ebird_classid_mapping.json")
        if not os.path.exists(map_path):
            self._ebird_cls_map = {}
            return self._ebird_cls_map

        try:
            import json
            with open(map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)  # {str(class_id): ebird_code}
            self._ebird_cls_map = {v: int(k) for k, v in raw.items()}
        except Exception as e:
            print(_t("logs.avonet_classid_failed", e=e))
            self._ebird_cls_map = {}

        return self._ebird_cls_map

    def _detect_country_from_gps(self, lat: float, lon: float) -> Optional[str]:
        """
        根据 GPS 坐标离线判定国家代码（仅返回国家级，不含州级）。
        优先匹配面积最小的边界框，避免大国遮蔽小国。
        """
        # 大陆级/全球代码，跳过
        _SKIP = {"GLOBAL", "AF", "AS", "EU", "NA", "SA", "OC"}

        # 收集匹配的国家及其面积
        candidates = []
        for code, bounds in REGION_BOUNDS.items():
            if code in _SKIP:
                continue
            south, north, west, east = bounds
            if south <= lat <= north and west <= lon <= east:
                area = (north - south) * (east - west)
                candidates.append((area, code))

        if not candidates:
            return None

        # 返回面积最小的匹配（最具体的）
        candidates.sort()
        return candidates[0][1]

    def get_species_by_country_ebird(
        self, lat: float, lon: float
    ) -> Tuple[Set[int], Optional[str]]:
        """
        根据 GPS 坐标判定国家，加载 eBird 离线物种列表，返回 class_id 集合。

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            (class_id_set, country_code) 或 (set(), None)
        """
        country_code = self._detect_country_from_gps(lat, lon)
        if not country_code:
            return set(), None

        # 加载对应国家的 eBird 物种列表
        module_dir = os.path.dirname(os.path.abspath(__file__))
        species_file = os.path.join(
            module_dir, "data", "offline_ebird_data",
            f"species_list_{country_code}.json"
        )
        if not os.path.exists(species_file):
            print(_t("logs.avonet_no_ebird_data", code=country_code))
            return set(), None

        try:
            import json
            with open(species_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ebird_codes: List[str] = data.get("species", [])
        except Exception as e:
            print(_t("logs.avonet_read_ebird_failed", code=country_code, e=e))
            return set(), None

        # 转换 eBird 代码 -> class_id
        cls_map = self._load_ebird_cls_map()
        class_ids: Set[int] = set()
        for code in ebird_codes:
            cls_id = cls_map.get(code)
            if cls_id is not None:
                class_ids.add(cls_id)

        return class_ids, country_code

    def get_species_by_region_ebird(
        self, region_code: str
    ) -> Tuple[Set[int], Optional[str]]:
        """
        根据州/省代码（如 "AU-QLD", "US-CA", "CN-44"）加载 eBird 离线物种列表。
        如果州级数据不存在，自动回退到国家级数据。

        Returns:
            (class_id_set, actual_region_used) 或 (set(), None)
        """
        import json
        module_dir = os.path.dirname(os.path.abspath(__file__))
        offline_dir = os.path.join(module_dir, "data", "offline_ebird_data")

        def _load_ebird_file(code: str) -> Optional[List[str]]:
            path = os.path.join(offline_dir, f"species_list_{code}.json")
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 支持两种格式：纯 list 或 {"species": [...]}
                if isinstance(data, list):
                    return data
                return data.get("species", [])
            except Exception as e:
                print(_t("logs.avonet_read_ebird_failed", code=code, e=e))
                return None

        region_code = region_code.upper()
        # 尝试加载州级数据
        species_codes = _load_ebird_file(region_code)
        actual_region = region_code

        # 州级无数据则回退到国家级
        if not species_codes and "-" in region_code:
            country = region_code.split("-")[0]
            species_codes = _load_ebird_file(country)
            actual_region = country if species_codes else None

        if not species_codes:
            return set(), None

        cls_map = self._load_ebird_cls_map()
        class_ids: Set[int] = set()
        for code in species_codes:
            cls_id = cls_map.get(code)
            if cls_id is not None:
                class_ids.add(cls_id)

        return class_ids, actual_region


if __name__ == "__main__":
    print("=" * 60)
    print("AvonetFilter 测试")
    print("=" * 60)

    # 创建过滤器实例
    af = AvonetFilter()

    # 检查数据库是否可用
    print(f"\n数据库路径: {af.db_path}")
    print(f"数据库可用: {af.is_available()}")

    if not af.is_available():
        print("错误: 数据库不可用，无法继续测试")
        exit(1)

    # 测试 GPS 查询
    print("\n" + "-" * 40)
    print("GPS 坐标查询测试")
    print("-" * 40)

    test_locations = [
        ("吉隆坡 (马来西亚)", 3.0, 101.7),
        ("悉尼 (澳大利亚)", -33.9, 151.2),
        ("东京 (日本)", 35.7, 139.7),
        ("伦敦 (英国)", 51.5, -0.1),
    ]

    for name, lat, lon in test_locations:
        species = af.get_species_by_gps(lat, lon)
        print(f"  {name}: {len(species)} 个物种")
        if species:
            sample = sorted(list(species))[:5]
            print(f"    样例 class_ids: {sample}")

    # 测试区域查询
    print("\n" + "-" * 40)
    print("区域代码查询测试")
    print("-" * 40)

    test_regions = ["AU", "AU-SA", "CN", "JP"]

    for region in test_regions:
        species = af.get_species_by_region(region)
        bounds = af.get_region_bounds(region)
        print(f"  {region}: {len(species)} 个物种")
        print(f"    边界: {bounds}")
        if species:
            sample = sorted(list(species))[:5]
            print(f"    样例 class_ids: {sample}")

    # 显示支持的区域列表
    print("\n" + "-" * 40)
    print("支持的区域代码")
    print("-" * 40)

    regions = af.get_supported_regions()
    print(f"  共 {len(regions)} 个区域:")

    # 按类别分组显示
    global_regions = [r for r in regions if r == "GLOBAL"]
    au_states = [r for r in regions if r.startswith("AU-")]
    au_country = [r for r in regions if r == "AU"]
    asia = [r for r in regions if r in ["CN", "JP", "KR", "TW", "TH", "MY", "SG", "ID", "PH", "VN", "IN", "NZ"]]
    americas = [r for r in regions if r in ["US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "EC", "CR"]]
    europe = [r for r in regions if r in ["GB", "FR", "DE", "ES", "IT", "NO", "SE", "FI", "PL", "TR"]]
    africa = [r for r in regions if r in ["ZA", "KE", "TZ", "EG", "MA"]]

    print(f"  全球: {global_regions}")
    print(f"  澳大利亚: {au_country + au_states}")
    print(f"  亚太: {asia}")
    print(f"  美洲: {americas}")
    print(f"  欧洲: {europe}")
    print(f"  非洲: {africa}")

    # 关闭连接
    af.close()
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
