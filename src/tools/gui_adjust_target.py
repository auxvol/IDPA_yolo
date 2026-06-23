import cv2
import numpy as np
import os
import sys

# Default normalized data (0~1) copied from idpa_scoring.py
cx = 0.504427
DATA = {
    "head_0": {'cx': cx, 'cy': 0.1097, 'rx': 0.1103, 'ry': 0.0650},
    "body_0": {'cx': cx, 'cy': 0.4024, 'rx': 0.2207, 'ry': 0.1301},
    "head_1": [
        [0.3345, 0.0122], [0.6655, 0.0122], [0.6655, 0.2073], [0.3345, 0.2073]
    ],
    "body_1": [
        [0.3345, 0.2073], [0.6655, 0.2073], [0.8310, 0.2588], [0.8310, 0.6094],
        [0.6655, 0.6951], [0.3345, 0.6951], [0.1690, 0.6094], [0.1690, 0.2588]
    ],
    "boundary_3": [
        [0.329828, 0.000000], [0.679026, 0.000000], [0.679026, 0.203805],
        [0.892086, 0.208972], [1.000000, 0.278788], [0.999447, 0.810260],
        [0.827892, 0.996325], [0.173215, 1.000000], [0.000000, 0.810260],
        [0.008301, 0.278788], [0.116215, 0.206634], [0.329828, 0.203805]
    ]
}

# State
dragging_node = None
mouse_pos = (0, 0)

# Window settings
WIN_NAME = "IDPA Target Adjuster"
DRAG_RADIUS = 8

def get_nodes(w, h):
    """Generate interactive nodes from DATA."""
    nodes = []
    
    # Polygons
    for key in ["boundary_3", "head_1", "body_1"]:
        pts = DATA[key]
        for i, pt in enumerate(pts):
            px, py = int(pt[0] * w), int(pt[1] * h)
            nodes.append({
                "type": "poly", "key": key, "idx": i,
                "px": px, "py": py
            })
            
    # Ellipses
    for key in ["head_0", "body_0"]:
        e = DATA[key]
        cx_px, cy_px = int(e['cx'] * w), int(e['cy'] * h)
        rx_px, ry_px = int(e['rx'] * w), int(e['ry'] * h)
        
        # Center point
        nodes.append({"type": "ellipse_center", "key": key, "px": cx_px, "py": cy_px})
        # Right edge (controls rx)
        nodes.append({"type": "ellipse_rx", "key": key, "px": cx_px + rx_px, "py": cy_px})
        # Bottom edge (controls ry)
        nodes.append({"type": "ellipse_ry", "key": key, "px": cx_px, "py": cy_px + ry_px})
        
    return nodes

def draw(img_bg):
    h, w = img_bg.shape[:2]
    img = img_bg.copy()
    
    # Draw -3
    pts3 = np.array(DATA["boundary_3"]) * [w, h]
    cv2.polylines(img, [pts3.astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)
    
    # Draw -1
    ptsh1 = np.array(DATA["head_1"]) * [w, h]
    cv2.polylines(img, [ptsh1.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    ptsb1 = np.array(DATA["body_1"]) * [w, h]
    cv2.polylines(img, [ptsb1.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Draw -0
    for k in ["head_0", "body_0"]:
        e = DATA[k]
        cv2.ellipse(img, 
                    (int(e['cx']*w), int(e['cy']*h)), 
                    (int(e['rx']*w), int(e['ry']*h)), 
                    0, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Draw Nodes
    nodes = get_nodes(w, h)
    for n in nodes:
        color = (0, 0, 255)
        if dragging_node and dragging_node['key'] == n['key']:
            if dragging_node.get('idx') == n.get('idx') and dragging_node['type'] == n['type']:
                color = (255, 255, 255) # Highlight dragging node
        cv2.circle(img, (n['px'], n['py']), 5, color, -1)
        cv2.circle(img, (n['px'], n['py']), DRAG_RADIUS, (255, 255, 255), 1)

    # Info text
    cv2.putText(img, "Drag red dots to adjust.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Press 's' to export Python code.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Press 'q' to quit.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow(WIN_NAME, img)

def mouse_cb(event, x, y, flags, param):
    global dragging_node, mouse_pos
    w, h = param['w'], param['h']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        nodes = get_nodes(w, h)
        # Find closest node
        best_n = None
        best_d = float('inf')
        for n in nodes:
            d = np.hypot(x - n['px'], y - n['py'])
            if d < DRAG_RADIUS * 1.5 and d < best_d:
                best_d = d
                best_n = n
        if best_n:
            dragging_node = best_n
            
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        if dragging_node:
            nx, ny = max(0, min(1.0, x / w)), max(0, min(1.0, y / h))
            
            if dragging_node["type"] == "poly":
                DATA[dragging_node["key"]][dragging_node["idx"]] = [nx, ny]
            elif dragging_node["type"] == "ellipse_center":
                DATA[dragging_node["key"]]['cx'] = nx
                DATA[dragging_node["key"]]['cy'] = ny
            elif dragging_node["type"] == "ellipse_rx":
                cx = DATA[dragging_node["key"]]['cx']
                DATA[dragging_node["key"]]['rx'] = abs(nx - cx)
            elif dragging_node["type"] == "ellipse_ry":
                cy = DATA[dragging_node["key"]]['cy']
                DATA[dragging_node["key"]]['ry'] = abs(ny - cy)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_node = None

def export_code():
    code = f"""
        # 以 0~1 歸一化座標定義
        # -0 區
        self.head_0 = {{'cx': {DATA['head_0']['cx']:.4f}, 'cy': {DATA['head_0']['cy']:.4f}, 'rx': {DATA['head_0']['rx']:.4f}, 'ry': {DATA['head_0']['ry']:.4f}}}
        self.body_0 = {{'cx': {DATA['body_0']['cx']:.4f}, 'cy': {DATA['body_0']['cy']:.4f}, 'rx': {DATA['body_0']['rx']:.4f}, 'ry': {DATA['body_0']['ry']:.4f}}}
        
        # -1 區
        self.head_1 = [
{chr(10).join(['            (' + ', '.join(f'{v:.4f}' for v in pt) + '),' for pt in DATA['head_1']])}
        ]
        self.body_1 = [
{chr(10).join(['            (' + ', '.join(f'{v:.4f}' for v in pt) + '),' for pt in DATA['body_1']])}
        ]
        
        # -3 區 (外框)
        self.boundary_3 = [
{chr(10).join(['            (' + ', '.join(f'{v:.6f}' for v in pt) + '),' for pt in DATA['boundary_3']])}
        ]
"""
    print("="*60)
    print("請將以下程式碼貼到 idpa_scoring.py 的 IDPATarget.__init__ 中：")
    print("="*60)
    print(code)
    print("="*60)

def main():
    img_path = r"c:\local_python\3_2\IDPA_yolo\data\idpa\idpa_target.png"
    if not os.path.exists(img_path):
        print(f"找不到圖片: {img_path}")
        # Create a blank image if not found
        img_bg = np.zeros((800, 500, 3), dtype=np.uint8)
        img_bg[:] = (100, 100, 100)
    else:
        img_bg = cv2.imread(img_path)
        
    h, w = img_bg.shape[:2]
    
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_cb, {"w": w, "h": h})
    
    while True:
        draw(img_bg)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            export_code()
            print("已經輸出程式碼！")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
