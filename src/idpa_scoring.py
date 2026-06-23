import math
from typing import List, Tuple, Dict, Union

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to determine if a point is inside a polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def point_in_ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float) -> bool:
    return ((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1

class IDPATarget:
    def __init__(self, target_width_pixels: float = None, target_height_pixels: float = None):
        """
        初始化 IDPA 靶紙的幾何定義。
        使用 0~1 的歸一化座標對齊 YOLO Pose 輸出的攤平畫面，確保外框完全吻合。
        """
        self.scale_x = target_width_pixels if target_width_pixels else 1.0
        self.scale_y = target_height_pixels if target_height_pixels else 1.0

        # 以 0~1 歸一化座標定義
        # -0 區
        self.head_0 = {'cx': 0.4982, 'cy': 0.0835, 'rx': 0.1326, 'ry': 0.0792}
        self.body_0 = {'cx': 0.5018, 'cy': 0.4497, 'rx': 0.2330, 'ry': 0.1499}
        
        # -1 區
        self.head_1 = [
            (0.3226, 0.0000), (0.6774, 0.0000), (0.6738, 0.2077), (0.3262, 0.2077)
        ]
        self.body_1 = [
            (0.3345, 0.2073), (0.6655, 0.2073), (0.8387, 0.2827), (0.8351, 0.6317),
            (0.7133, 0.8094), (0.2867, 0.8073), (0.1613, 0.6210), (0.1685, 0.2762)
        ]
        
        # -3 區 (外框，完全使用 NORMALIZED_POINTS_17 的 12 個外圍點確保對齊)
        self.boundary_3 = [
            (0.329828, 0.000000), # 1: 左頭頂
            (0.679026, 0.000000), # 2: 右頭頂
            (0.679026, 0.203805), # 3: 右頸部
            (0.892086, 0.208972), # 4: 右肩內側
            (1.000000, 0.278788), # 5: 右肩外緣
            (0.999447, 0.810260), # 6: 右腰部
            (0.827892, 0.996325), # 7: 右底角
            (0.173215, 1.000000), # 8: 左底角
            (0.000000, 0.810260), # 9: 左腰部
            (0.008301, 0.278788), # 10: 左肩外緣
            (0.116215, 0.206634), # 11: 左肩內側
            (0.329828, 0.203805), # 0: 左頸部
        ]
        
        # 套用縮放
        self._scale_ellipse(self.head_0)
        self._scale_ellipse(self.body_0)
        self.head_1 = self._scale_polygon(self.head_1)
        self.body_1 = self._scale_polygon(self.body_1)
        self.boundary_3 = self._scale_polygon(self.boundary_3)

    def _scale_ellipse(self, c: Dict):
        c['cx'] *= self.scale_x
        c['cy'] *= self.scale_y
        c['rx'] *= self.scale_x
        c['ry'] *= self.scale_y

    def _scale_polygon(self, poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(x * self.scale_x, y * self.scale_y) for x, y in poly]

    def get_hit_zone(self, x: float, y: float) -> Union[int, str]:
        """
        根據給定的座標，判斷命中哪個計分區。
        回傳值: 0, 1, 3 或 "Miss"
        """
        # 優先判定 -0 區
        if point_in_ellipse(x, y, self.head_0['cx'], self.head_0['cy'], self.head_0['rx'], self.head_0['ry']):
            return 0
        if point_in_ellipse(x, y, self.body_0['cx'], self.body_0['cy'], self.body_0['rx'], self.body_0['ry']):
            return 0
            
        # 判定 -1 區
        if point_in_polygon(x, y, self.head_1) or point_in_polygon(x, y, self.body_1):
            return 1
            
        # 判定 -3 區
        if point_in_polygon(x, y, self.boundary_3):
            return 3
            
        # 落於靶紙外
        return "Miss"


class IDPAScorer:
    def __init__(self, target_width_pixels: float = None, target_height_pixels: float = None):
        """
        初始化 IDPA 計分器。
        可以透過提供 target_width_pixels 與 target_height_pixels 
        來設定與偵測模型對齊的座標系 (例如 640x640 等)。
        """
        self.target = IDPATarget(target_width_pixels, target_height_pixels)
        
    def score_single_target(self, hits: List[Tuple[float, float]], shots_required: int = 2) -> Dict:
        """
        計算單一靶紙的成績。
         hits: 命中點的座標列表，例如 [(x1, y1), (x2, y2), ...]
         shots_required: 該靶需要計算的最佳發數，預設為 2 發
        """
        hit_zones = []
        for x, y in hits:
            zone = self.target.get_hit_zone(x, y)
            hit_zones.append(zone)
            
        # 轉換成罰秒點數 (Points Down: -0 -> 0s, -1 -> 1s, -3 -> 3s, Miss -> 5s)
        points_down_list = []
        for zone in hit_zones:
            if zone == 0: points_down_list.append(0)
            elif zone == 1: points_down_list.append(1)
            elif zone == 3: points_down_list.append(3)
            elif zone == "Miss": points_down_list.append(5)
            
        # 排序以取得最佳的擊發 (罰秒最少的)
        points_down_list.sort()
        
        # 擷取最佳的 shots_required 發
        best_hits = points_down_list[:shots_required]
        
        # 如果實際擊發數少於要求發數，補上 Miss (5 秒)
        misses_to_add = max(0, shots_required - len(best_hits))
        best_hits.extend([5] * misses_to_add)
        
        # 計算單靶總罰秒
        total_target_penalty = sum(best_hits)
        
        return {
            'hit_zones_raw': hit_zones,              # 每一發落點區域
            'points_down_raw': points_down_list,     # 每一發轉換的罰秒
            'best_hits_points_down': best_hits,      # 最佳發數的罰秒列表
            'total_target_penalty': total_target_penalty # 單靶總罰秒 (秒)
        }

    def calculate_stage_score(self, raw_time: float, targets_hits: List[List[Tuple[float, float]]], 
                              shots_per_target: int = 2, procedural_errors: int = 0, 
                              hits_on_non_threat: int = 0) -> Dict:
        """
        計算整個 Stage 的總成績。
        
        raw_time: 基礎時間（從計時器到最後一槍發射的時間）
        targets_hits: 每個靶紙的彈孔座標。範例: [[(x,y), (x,y)], [(x,y)], ...]
        shots_per_target: 每靶要求發數 (預設 2 發)
        procedural_errors (PE): 程序錯誤次數 (每次加 3 秒)
        hits_on_non_threat (HNT): 誤擊人質靶次數 (每次加 5 秒)
        """
        total_points_down_penalty = 0
        target_results = []
        
        for idx, hits in enumerate(targets_hits):
            res = self.score_single_target(hits, shots_required=shots_per_target)
            total_points_down_penalty += res['total_target_penalty']
            target_results.append({
                'target_id': idx + 1,
                'details': res
            })
            
        # 計算其他罰秒
        pe_penalty = procedural_errors * 3.0
        hnt_penalty = hits_on_non_threat * 5.0
        
        # 總成績 = 基礎時間 + 靶紙扣秒 + 犯規罰秒
        final_score = raw_time + total_points_down_penalty + pe_penalty + hnt_penalty
        
        return {
            'raw_time': raw_time,
            'points_down_penalty': total_points_down_penalty,
            'procedural_error_penalty': pe_penalty,
            'hnt_penalty': hnt_penalty,
            'final_score': final_score,
            'target_results': target_results
        }

if __name__ == "__main__":
    # 測試範例：
    # 假設我們將靶紙映射到 640x640 的圖片上（註：IDPA 靶紙不是正方形，
    # 這裡只是示範若直接使用圖片的像素長寬如何初始化）
    scorer = IDPAScorer(target_width_pixels=640, target_height_pixels=640)
    
    # 假設打靶1 命中 (中心, 邊緣) -> (-0, -3) -> 取兩發最佳 (0+3) = 3秒
    target_1_hits = [(320, 100), (600, 320)] 
    
    # 假設打靶2 命中 (中心, 中心, 錯失) -> (-0, -0, Miss) -> 取兩發最佳 (0+0) = 0秒
    target_2_hits = [(320, 260), (330, 270), (10, 10)] 
    
    stage_result = scorer.calculate_stage_score(
        raw_time=12.50, 
        targets_hits=[target_1_hits, target_2_hits],
        shots_per_target=2,
        procedural_errors=0,
        hits_on_non_threat=0
    )
    
    print(f"Base Time: {stage_result['raw_time']} s")
    print(f"Points Down Penalty: +{stage_result['points_down_penalty']} s")
    print(f"Total Stage Score: {stage_result['final_score']} s")
    print("\n詳細資訊:")
    for res in stage_result['target_results']:
        print(f"  Target {res['target_id']}: 區域={res['details']['hit_zones_raw']} -> 計分={res['details']['best_hits_points_down']}")
