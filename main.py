import cv2
import numpy as np

def find_icon(desktop_image_path, icon_template_path):
    # 读取桌面图片和图标模板
    desktop = cv2.imread(desktop_image_path)
    template = cv2.imread(icon_template_path)
    
    # 获取图片尺寸
    desktop_height, desktop_width = desktop.shape[:2]
    template_height, template_width = template.shape[:2]
    
    print(f"桌面图片尺寸: {desktop_width} x {desktop_height} 像素")
    print(f"图标模板尺寸: {template_width} x {template_height} 像素")

    # 执行模板匹配
    result = cv2.matchTemplate(desktop, template, cv2.TM_CCOEFF_NORMED)
    
    # 提高阈值，使匹配更严格
    threshold = 0.9  # 提高阈值
    locations = np.where(result >= threshold)
    
    # 获取所有匹配位置
    matches = []
    for pt in zip(*locations[::-1]):
        x1, y1 = pt
        x2, y2 = (x1 + template_width, y1 + template_height)
        matches.append([x1, y1, x2, y2])
    
    # 使用非极大值抑制(NMS)去除重叠的检测框
    matches = np.array(matches)
    if len(matches) > 0:
        # 转换为 [x1, y1, x2, y2] 格式
        boxes = matches
        
        # 计算所有框的面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 根据置信度排序（这里我们用y坐标作为排序依据）
        order = boxes[:, 1].argsort()
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算当前框与其他框的IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的框
            inds = np.where(ovr <= 0.3)[0]
            order = order[inds + 1]
        
        matches = boxes[keep]
    
    # 转换结果格式
    final_matches = []
    for x1, y1, x2, y2 in matches:
        final_matches.append({
            "左上角": (int(x1), int(y1)),
            "右下角": (int(x2), int(y2)),
            "中心点": (int((x1 + x2)//2), int((y1 + y2)//2)),
            "宽度": int(x2 - x1),
            "高度": int(y2 - y1),
            "相对位置_左上": (f"{int(x1/desktop_width*100)}%", f"{int(y1/desktop_height*100)}%"),
            "相对位置_右下": (f"{int(x2/desktop_width*100)}%", f"{int(y2/desktop_height*100)}%")
        })
    
    return final_matches

# 使用函数
desktop_path = "desktop.png"
icon_path = "chrome.png"

matches = find_icon(desktop_path, icon_path)

# 打印结果
if matches:
    print(f"\n找到 {len(matches)} 个匹配的微信图标:")
    for i, match in enumerate(matches, 1):
        print(f"\n图标 #{i}:")
        print(f"  位置: 左上角{match['左上角']}, 右下角{match['右下角']}")
        print(f"  中心点: {match['中心点']}")
        print(f"  尺寸: {match['宽度']} x {match['高度']} 像素")
        print(f"  相对位置: 左上角{match['相对位置_左上']}, 右下角{match['相对位置_右下']}")
else:
    print("未找到微信图标")
