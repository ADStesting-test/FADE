import numpy as np
from PIL import Image, ImageDraw
import random
from scipy.spatial import ConvexHull

def add_circle(image):
    """
    在图像上添加一个圆,图像是 PIL的Image对象
    """

    draw = ImageDraw.Draw(image)
    # 图像尺寸
    width, height = image.size

    # 随机中心点
    x, y = random.randint(0, width), random.randint(0, height) 
    # 随机 半径
    radius = random.randint(10, min(width, height) // 4)
    # 填充，黑色
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='black')

    return image

def add_rectangle(image):
    """
    在图像上添加一个矩形,图像是 PIL的Image对象
    """

    draw = ImageDraw.Draw(image)
    # 图像尺寸
    width, height = image.size

    # 随机 左上角的点
    x, y = random.randint(0, width), random.randint(0, height)
    # 随机 宽高
    w, h = random.randint(10, width // 4), random.randint(10, height // 4)
    # 黑色填充
    draw.rectangle((x, y, x + w, y + h), fill='black')

    return image

def add_polygon(image):
    """
    在图像上添加一个 多边形,图像是 PIL的Image对象
    """

    draw = ImageDraw.Draw(image)
    # 图像尺寸
    width, height = image.size
    # 随机 顶点
    points = [(random.randint(0, width), random.randint(0, height)) for _ in range(5)]
    # 黑色填充
    draw.polygon(points, fill='black')

    return image

def add_irregular_patch(image,num_points=30,size_factor=0.2,color = False ):
    """
    在图像上添加一个 不规则图形,图像是 PIL的Image对象
    num_points: 多边形顶点数
    size_factor: 多边形最大尺寸与图像尺寸的比值
    """
    # 设置种子值
    # random.seed(42)
    draw = ImageDraw.Draw(image)
    # 图像尺寸
    width, height = image.size

    # 随机添加顶点
    points = []
    for _ in range(num_points):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        points.append((x, y))

    # 获取多边形外部的顶点（凸包）
    hull = ConvexHull(points)
    points = [points[vertex] for vertex in hull.vertices]
    
    # 根据大小因子调整顶点位置
    center_x, center_y = width / 2, height / 2
    for i in range(len(points)):
        points[i] = ((points[i][0] - center_x) * size_factor + center_x, (points[i][1] - center_y) * size_factor + center_y)
    
    # 计算形状的边界框
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    
    # 随机偏移边界框的位置
    offset_x = random.uniform(-min_x, width - max_x)
    offset_y = random.uniform(-min_y, height - max_y)
    
    # 应用偏移量到所有顶点
    new_points = [(point[0] + offset_x, point[1] + offset_y) for point in points]
    
    if color :
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        draw.polygon(new_points, fill=(r, g, b))
    else :
        draw.polygon(new_points, fill='black')

    return image