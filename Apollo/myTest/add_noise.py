import occlusion
import numpy as np
import cv2
import copy
from PIL import Image, ImageDraw,ImageOps,ImageEnhance,ImageFilter
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from io import BytesIO
import math

out_folder = "/apollo/data/image/"

# 将PIL图像转换为NumPy数组
def pil_to_array(img):
    return np.asarray(img)

# 将NumPy数组转换为PIL图像
def array_to_pil(arr):
    return Image.fromarray(np.uint8(arr))

def add_gaussian_noise(image, mean=0, sigma=30):
    """给图片添加高斯噪声"""
    # 将PIL图像转换为NumPy数组
    img_array = pil_to_array(image)
    
    # 生成高斯噪声
    noise = np.random.normal(mean, sigma, img_array.shape)
    
    # 将噪声添加到图像
    noisy_image = img_array + noise
    
    # 将图像裁剪到合法范围 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # 将结果转换回PIL图像
    return array_to_pil(noisy_image)

# 可以通过 OpenCV 的高斯模糊函数实现
def add_blur(image, blur_strength=15):
    """对图像应用高斯模糊"""
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

# 可以在图像的随机区域上添加矩形或其他形状进行遮挡
def add_occlusion(image, size=50,num_occ = 1,occ_type="irregular"):

    """在图像上添加随机遮挡效果"""
    if(num_occ > 1):
        size_factor = 0.02
        color = True
    else:
        size_factor = 0.2
        color = False

    for i in range(num_occ):
        if(occ_type=="circle"):
            image = occlusion.add_circle(image)
        elif(occ_type=="rectangle"):
            image = occlusion.add_rectangle(image)
        elif(occ_type=="polygon"):
            image = occlusion.add_polygon(image)
        elif(occ_type=="irregular"):
            image = occlusion.add_irregular_patch(image,num_points=size,size_factor = size_factor,color=color)
        else:
            print("no occ_type"+occ_type)
            break
    return image


# 可以通过在图像上添加随机的点来模拟散点分布
def add_scatter(image, num_points=10000, distribution='uniform', center=(0, 0), sigma=0.1):
    """在图像上添加散点效果"""
     # 确保图像是 RGBA 模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(num_points):
        if distribution == 'uniform':
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
        elif distribution == 'gaussian':
            x = int(np.random.normal(center[0], sigma * width))
            y = int(np.random.normal(center[1], sigma * height))
            x = np.clip(x, 0, width - 1)  # 限制范围
            y = np.clip(y, 0, height - 1)

        # 随机设置散点颜色、大小和透明度
        color = np.random.randint(0, 256, size=3)  # 随机颜色
        alpha = np.random.randint(50, 256)  # 随机透明度
        radius = np.random.randint(1, 4)  # 随机半径

        # 创建带透明度的颜色
        rgba_color = (*color, alpha)

        # 绘制散点
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba_color)
    if image.mode == 'RGBA':
        # Convert to RGB
        image = image.convert('RGB')

    return image

# 可以通过在图像上添加随机的白色条纹来实现 雨水效果
def add_rain_effect(image, num_drops=500, drop_length=20, wind_strength=45, drop_thickness=2, alpha_min=0.1, alpha_max=0.6):
    """
    在图像上添加模拟雨水效果。
    
    参数:
    - image: PIL的Image对象
    - num_drops: 雨滴的数量
    - drop_length: 每个雨滴的长度
    - wind_strength: 模拟风的强度，影响雨滴的倾斜角度
    - drop_thickness: 雨滴的厚度
    - alpha_min/max: 雨滴的透明度范围
    
    返回:
    - 添加雨水效果后的图像(PIL Image)
    """
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # 创建一个用于绘制雨滴的透明层
    rain_layer = np.zeros((h, w, 4), dtype=np.uint8)

    # 随机确定雨滴的倾斜角度（受风影响）
    wind_offset = np.random.randint(-wind_strength, wind_strength)
    for _ in range(num_drops):
        # 随机确定雨滴的起点
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        
        # 随机确定雨滴的长度
        length = np.random.randint(10, drop_length)
        
        # 确定雨滴的终点
        x_end = x + wind_offset
        y_end = y + length
        
        # 随机确定雨滴的透明度
        alpha = np.random.uniform(alpha_min, alpha_max)
        
        # 设置雨滴的颜色（白色）和透明度
        color = (255, 255, 255, int(alpha * 255))
        
        # 绘制雨滴
        cv2.line(rain_layer, (x, y), (x_end, y_end), color, drop_thickness)

    # 将雨滴层与原图像进行叠加
    overlay_img = np.concatenate([img_array, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)  # 扩展成4通道
    combined = cv2.addWeighted(overlay_img, 1, rain_layer, 0.5, 0)
    
    # 转换回 PIL 图像格式
    return Image.fromarray(combined[:, :, :3].astype(np.uint8))

def add_snow_effect(image, num_snow=1000, size_range=(1, 5), alpha_range=(50, 150), light_position=None):
    """
    在图像中添加灰尘效果。

    参数:
    - image: PIL的Image对象
    - num_snow: 灰尘粒子的数量
    - size_range: 灰尘粒子的大小范围（最小和最大半径）
    - alpha_range: 灰尘粒子的透明度范围(0-255)
    - light_position: 光源的位置（可选），用于调整灰尘粒子的亮度

    返回:
    - 添加灰尘效果后的图像(PIL Image)
    """
    img_array = np.array(image)
    h, w, _ = img_array.shape

    # 创建一个新的图像用于绘制灰尘
    snow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(snow_layer)

    for _ in range(num_snow):
        # 随机生成灰尘粒子的属性
        radius = np.random.randint(size_range[0], size_range[1])
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        alpha = np.random.randint(alpha_range[0], alpha_range[1])

        # 设置灰尘粒子的颜色（灰色）
        color = (150, 150, 150, alpha)

        # 计算光照影响
        if light_position is not None:
            # 光源位置
            x_L, y_L = light_position
            d = np.sqrt((x - x_L)**2 + (y - y_L)**2)  # 到光源的距离
            light_intensity = 255 / (1 + d)  # 光照强度衰减
            color = (int(color[0] *(1+ light_intensity )), int(color[1] * (1+light_intensity )), int(color[2]*(1+light_intensity)), alpha)

        # 绘制灰尘粒子
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    # 将灰尘层与原图像进行合成
    snow_array = np.array(snow_layer)
    combined = Image.alpha_composite(Image.fromarray(img_array).convert("RGBA"), snow_layer)

    return combined.convert("RGB")

# 定义过度曝光模型函数
def apply_overexposure_pil(image, light_position, light_intensity=220, radius=500):
    img_array = np.asarray(image)
    
    # 获取图像的宽度和高度
    height, width, _ = img_array.shape
    
    # 创建一个光源影响的高斯分布
    y, x = np.ogrid[:height, :width]
    distance_from_light = np.sqrt((x - light_position[0])**2 + (y - light_position[1])**2)
    
    # 使用高斯函数模拟光源影响 (可以调整sigma值来控制光源扩散范围)
    gaussian_light = np.exp(-(distance_from_light**2 / (2.0 * radius**2)))
    
    # 模拟光源的强度并添加到图像上
    light_effect = light_intensity * gaussian_light[..., np.newaxis]  # 增加维度以匹配RGB图像
    
    # 将光效添加到原始图像并限制在255以内
    overexposed_img = np.clip(img_array + light_effect, 0, 255)
    
    # 将结果转换回PIL图像
    return Image.fromarray(np.uint8(overexposed_img))

    # 定义白平衡偏移函数
def apply_white_balance_shift(image, red_balance=1.75, green_balance=1.75, blue_balance=0.5):
    # 将PIL图像转换为NumPy数组
    img_array = pil_to_array(image)
    
    # 分离RGB通道
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    # 应用白平衡调整
    red_channel = np.clip(red_channel * red_balance, 0, 255)
    green_channel = np.clip(green_channel * green_balance, 0, 255)
    blue_channel = np.clip(blue_channel * blue_balance, 0, 255)
    
    # 将调整后的通道组合回图像
    balanced_img = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    
    # 将结果转换回PIL图像
    return array_to_pil(balanced_img)
def generate_crack_params(center, max_radius=30, num_cracks=30):
    """生成裂缝参数，包括角度、扩展长度、宽度和透明度"""
    cracks = []
    for _ in range(num_cracks):
        angle = random.uniform(0, 2 * np.pi)  # 随机角度
        radius = random.randint(10, max_radius)  # 随机半径
        delta_r = random.randint(100, 1000)  # 随机扩展长度
        width = int(np.random.normal(2, 0.5))  # 裂缝宽度，正态分布
        alpha = random.randint(100, 200)  # 裂缝透明度
        
        cracks.append((angle, radius, delta_r, width, alpha))
    
    return cracks

def add_cracks(image, center, max_radius=100, num_cracks=100):
    """在图像上添加从中心向外扩散的裂缝效果"""
    draw = ImageDraw.Draw(image)
    cracks = generate_crack_params(center, max_radius, num_cracks)
    
    for angle, radius, delta_r, width, alpha in cracks:
        # 计算起始和结束点
        start_x = int(center[0] + radius * np.cos(angle))
        start_y = int(center[1] + radius * np.sin(angle))
        end_x = int(center[0] + (radius + delta_r) * np.cos(angle))
        end_y = int(center[1] + (radius + delta_r) * np.sin(angle))

        # 裂缝颜色（白色，带透明度）
        crack_color = (255, 255, 255, alpha)
        
        # 绘制裂缝
        draw.line([(start_x, start_y), (end_x, end_y)], fill=crack_color, width=width)

    return image


# Convert PIL Image to OpenCV format
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV compatibility
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

# Convert OpenCV Image to PIL format
def cv2_to_pil(cv2_image):
    # Convert BGR to RGB for PIL compatibility
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image_rgb)

# Step 1: 创建随机种子点生成Voronoi图
def generate_voronoi( img_size,num_points):
    # 为每个维度分别生成随机坐标
    points = np.zeros((num_points, 2))
    points[:, 0] = np.random.rand(num_points) * img_size[0]  # X坐标
    points[:, 1] = np.random.rand(num_points) * img_size[1]  # Y坐标
    vor = Voronoi(points)
    return vor, points

# Step 2: 绘制Voronoi图并生成裂缝（Voronoi边界）
def draw_voronoi_cracks(image, vor):
    # 在白色背景图像上绘制Voronoi边界
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        pt1 = tuple(vor.vertices[ridge[0]].astype(int))
        pt2 = tuple(vor.vertices[ridge[1]].astype(int))
        # 在图像上画裂缝线
        cv2.line(image, pt1, pt2, (0, 0, 0), 1)
    return image

# Step 3: 添加Perlin噪声模拟裂缝扩展和细化
def perlin_noise(image, scale=10):
    noise = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            noise[i, j] = (random.random() - 0.5) * scale
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Step 4: 综合流程
def add_cracks_to_pil_image(pil_image):
    input_cv2_image = pil_to_cv2(pil_image)
    img_size = (input_cv2_image.shape[1],input_cv2_image.shape[0])
    # 创建白色背景
    cracks_image = input_cv2_image

    # Step 1: 生成Voronoi图
    num_points = 10  # 种子点数量
    vor, points = generate_voronoi(img_size,num_points)

    # Step 2: 生成裂缝
    cracks_image = draw_voronoi_cracks(cracks_image, vor)

    # Step 3: 添加Perlin噪声以模拟裂缝扩展
    noisy_cracks_image = perlin_noise(cracks_image)

    # 转换为PIL格式
    return cv2_to_pil(noisy_cracks_image)

def draw_zigzag_crack_with_quads_and_triangle(draw, start_point, crack_length, crack_width, crack_color):
    """Draws a zigzag crack divided into quads, with a sharp triangle at the end."""
    x1, y1 = start_point
    angle = random.uniform(0, 2 * np.pi)

    # Number of segments to break the crack into
    num_segments = random.randint(3, 5)
    segment_length = crack_length / num_segments

    current_width = crack_width
    

    for _ in range(num_segments - 1):
        # Calculate the next point
        x2 = x1 + segment_length * np.cos(angle)
        y2 = y1 + segment_length * np.sin(angle)

        # Adjust the direction slightly for the next segment
        angle += random.uniform(-np.pi / 6, np.pi / 6)

        # Calculate the four points of the quad (two for the previous width, two for the new width)
        next_width = current_width * random.uniform(0.4, 0.7)  # Gradually reduce width

        # Points for the previous segment (current width)
        base_left_x1 = x1 + current_width * np.cos(angle + np.pi / 2)
        base_left_y1 = y1 + current_width * np.sin(angle + np.pi / 2)

        base_right_x1 = x1 + current_width * np.cos(angle - np.pi / 2)
        base_right_y1 = y1 + current_width * np.sin(angle - np.pi / 2)

        # Points for the next segment (next width)
        base_left_x2 = x2 + next_width * np.cos(angle + np.pi / 2)
        base_left_y2 = y2 + next_width * np.sin(angle + np.pi / 2)

        base_right_x2 = x2 + next_width * np.cos(angle - np.pi / 2)
        base_right_y2 = y2 + next_width * np.sin(angle - np.pi / 2)

        # Draw the quad representing this segment of the crack
        draw.polygon([(base_left_x1, base_left_y1), (base_right_x1, base_right_y1), 
                      (base_right_x2, base_right_y2), (base_left_x2, base_left_y2)], 
                     fill=crack_color)

        # Move to the next point
        x1, y1 = x2, y2
        current_width = next_width

    # Draw the final segment as a sharp triangle
    final_x = x1 + segment_length * np.cos(angle)
    final_y = y1 + segment_length * np.sin(angle)

    # The base of the final triangle
    base_left_x = x1 + current_width * np.cos(angle + np.pi / 2)
    base_left_y = y1 + current_width * np.sin(angle + np.pi / 2)

    base_right_x = x1 + current_width * np.cos(angle - np.pi / 2)
    base_right_y = y1 + current_width * np.sin(angle - np.pi / 2)

    # Draw the triangle representing the sharp end
    draw.polygon([(final_x, final_y), (base_left_x, base_left_y), (base_right_x, base_right_y)], fill=crack_color)

def simulate_custom_zigzag_cracks(image, num_cracks=5, crack_color=(0, 0, 0), crack_width_range=(20, 30), crack_length_range=(200, 1000),num_sides_range=(20, 30), radius_range=(50, 200)):
    """
    Simulates cracks radiating outward from a center, with each crack divided into quads and ending with a sharp triangle.
    
    Args:
    - image: PIL Image object.
    - num_cracks: Number of cracks.
    - crack_color: Color of the cracks.
    - crack_width_range: Range for the starting width of the cracks.
    - crack_length_range: Range for the total length of each crack.
    
    Returns:
    - Image with simulated zigzag cracks.
    """
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Center of the image
    center_x, center_y = width // 2, height // 2

    # Generate an irregular polygon at the center
    num_sides = random.randint(num_sides_range[0], num_sides_range[1])
    # radius = random.randint(radius_range[0], radius_range[1])
    angle_step = 360 / num_sides

    polygon_points = []
    for i in range(num_sides):
        radius = random.randint(radius_range[0], radius_range[1])
        angle = angle_step * i + random.uniform(-10, 10)
        x = center_x + radius * np.cos(np.radians(angle)) + random.uniform(-5, 5)
        y = center_y + radius * np.sin(np.radians(angle)) + random.uniform(-5, 5)
        polygon_points.append((x, y))

    # Draw the irregular polygon to simulate the shattered center
    draw.polygon(polygon_points, fill=crack_color)

    # Generate main cracks radiating outward from the center
    for _ in range(num_cracks):
        angle = random.uniform(0, 2 * np.pi)  # Random angle
        length = random.randint(200, min(width, height))  # Random length for each crack
        end_x = center_x + length * np.cos(angle)
        end_y = center_y + length * np.sin(angle)
        main_width=random.randint(crack_width_range[0],crack_width_range[1])

        # Draw the triangular crack
        triangle_tip = (end_x, end_y)  # The sharp end point of the triangle
        triangle_base_left = (center_x - main_width, center_y - main_width)
        triangle_base_right = (center_x + main_width, center_y - main_width)
        draw.polygon([triangle_tip, triangle_base_left, triangle_base_right], fill=crack_color)

        # Generate a nested triangle at a random position on the sides of the main triangle
        random_position = random.uniform(0.5, 0.8)  # Random position along the side of the triangle
        base_left_x = triangle_base_left[0] + random_position * (triangle_tip[0] - triangle_base_left[0])
        base_left_y = triangle_base_left[1] + random_position * (triangle_tip[1] - triangle_base_left[1])
        base_left = (base_left_x,base_left_y)

        def point_on_line(base_left, triangle_base_left, length):
            # Get the coordinates of base_left and triangle_base_left
            x1, y1 = base_left
            x2, y2 = triangle_base_left
            
            # Calculate the direction vector
            v_x = x2 - x1
            v_y = y2 - y1
            
            # Calculate the distance between base_left and triangle_base_left
            distance = math.sqrt(v_x**2 + v_y**2)
            
            # Normalize the direction vector
            u_x = v_x / distance
            u_y = v_y / distance
            
            # Calculate the point that is 'length' distance away from base_left on the line
            x_p = x1 + u_x * length
            y_p = y1 + u_y * length
            
            return (x_p, y_p)

        base_right = point_on_line(base_left,triangle_base_left,length = main_width*(1-random_position)*2 )

        angle = angle+random.uniform(-np.pi/3, np.pi/3)  # Random angle
        nested_length = random.randint(50, 100)  # Random length for each crack
        nested_tip = (base_left_x + nested_length * np.cos(angle), base_left_y + nested_length * np.sin(angle))
        draw.polygon([nested_tip, base_left, base_right], fill=crack_color)

    # Generate cracks radiating outward
    for _ in range(num_cracks):
        crack_length = random.randint(*crack_length_range)
        crack_width = random.randint(*crack_width_range)
        
        # Draw zigzag cracks with quads and sharp ends
        draw_zigzag_crack_with_quads_and_triangle(draw, (center_x, center_y), crack_length, crack_width, crack_color)

    return image.convert("RGB")

def add_ice(image,alpha_factor=0.8):
    """添加冰冻效果"""
    # 打开冰霜效果图片和目标图片
    frost_path = '/apollo/modules/myTest/image/ice_origin2.jpeg'  # 冰霜效果图片路径
    frost_image = Image.open(frost_path).convert("RGBA")

    # 将冰霜效果图片调整为与原始图片相同的尺寸
    frost_image = frost_image.resize(image.size)

    # 转换为灰度图并应用阈值提取白色区域作为透明遮罩
    gray_frost = ImageOps.grayscale(frost_image)
    threshold = 200  # 设置阈值，将接近白色的区域提取出来
    mask = gray_frost.point(lambda x: 255 if x > threshold else 0).convert("L")

    # 创建带透明度的冰霜效果图层
    frost_with_alpha = Image.merge("RGBA", (frost_image.split()[0], frost_image.split()[1], frost_image.split()[2], mask))

    # 调整冰霜效果的透明度
    frost_with_alpha = frost_with_alpha.point(lambda p: p * alpha_factor if p < 255 else p)

    # 确保输入的原始图片也是 RGBA 模式
    image = image.convert("RGBA")

    # 使用 alpha_composite 叠加冰霜效果
    image = Image.alpha_composite(image, frost_with_alpha)
    image=image.convert("RGB")

    return image


def add_dust_effect(image, num_dust_spots=5, spot_size=(300, 500), brightness_factor=0.85):
    """
    在图像上添加灰尘效果，包括不规则的白色斑点和亮度调整。
    
    参数:
        image: 原始图片（PIL Image 对象）。
        num_dust_spots: 生成的灰尘斑点数量。
        spot_size: 灰尘斑点的大小范围。
        brightness_factor: 调整非斑点区域的整体亮度，<1 代表变暗。
    
    返回:
        叠加灰尘效果后的图片。
    """
    # 创建一个图层用于绘制灰尘斑点
    dust_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dust_layer)
    
    width, height = image.size

    # 生成随机的灰尘斑点
    for _ in range(num_dust_spots):
        # 随机选择灰尘斑点的位置、大小和亮度
        x, y = random.randint(0, width), random.randint(0, height)
        spot_w, spot_h = random.randint(*spot_size), random.randint(*spot_size)
        brightness = random.randint(200, 255)  # 接近白色

        # 绘制椭圆形状的灰尘斑点
        draw.ellipse([x, y, x + spot_w, y + spot_h], fill=(brightness, brightness, brightness, random.randint(100, 150)))

    # 对灰尘斑点进行模糊处理
    dust_layer = dust_layer.filter(ImageFilter.GaussianBlur(2))

    # 将灰尘斑点叠加到原始图片上
    image_with_dust = Image.alpha_composite(image.convert("RGBA"), dust_layer)

    # 对整体图片进行亮度调整（除了灰尘区域）
    enhancer = ImageEnhance.Brightness(image_with_dust)
    image_with_dust = enhancer.enhance(brightness_factor)

    return image_with_dust.convert("RGB")

def process_image_test(image,noisy_type,num=None,light_position=None):
    if noisy_type=="gaussian":
        # 添加高斯噪声
        noisy_image = add_gaussian_noise(image)
        # noisy_image = add_blur(image)
    elif noisy_type=="occlusion":
        if num:
            noisy_image = add_occlusion(image,num_occ=num)
        else:
            # 添加矩形或其他形状进行遮挡
            noisy_image = add_occlusion(image)
    elif noisy_type=="scatter":
        # 添加随机的白色或黑色小点来模拟散点分布
        noisy_image = add_scatter(image)
    elif noisy_type=="ice":
        noisy_image = add_ice(image)
    elif noisy_type=="rain_effect":
        # 添加随机的白色条纹来实现 雨水效果
        noisy_image= add_rain_effect(image)
    elif noisy_type=="dust_effect":
        noisy_image = add_dust_effect(image)
    elif noisy_type=="snow_effect":# 随机生成半透明的斑点来实现 雪花效果
        if light_position:
            img_array = np.asarray(image)
            height, width, _ = img_array.shape
            light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
            noisy_image = add_snow_effect(image,light_position=light_position)
        else:
            noisy_image = add_snow_effect(image)
    elif noisy_type=="overexposure":
        img_array = np.asarray(image)
        height, width, _ = img_array.shape
        light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
        noisy_image = apply_overexposure_pil(image,light_position)
    elif noisy_type=="white_balance":
        noisy_image = apply_white_balance_shift(image)
    elif noisy_type=="cracks":
        img_array = np.asarray(image)
        height, width, _ = img_array.shape
        light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
        if num:
            noisy_image = add_cracks(image,light_position,num_cracks = num)
        else:
            noisy_image = add_cracks(image,light_position)
    elif noisy_type=="large_cracks":
        noisy_image = add_cracks_to_pil_image(image)
    elif noisy_type=="radiating_cracks":
        noisy_image = simulate_custom_zigzag_cracks(image)
    else :
        print("noisy_type not find!!!")
        return image
    
    return noisy_image



def process_image(compressed_image,noisy_type,num=None,light_position=None):
    """处理 CompressedImage 消息，添加噪声并返回"""

    # 解码压缩图像数据为 OpenCV 图像格式
    np_arr = np.frombuffer(compressed_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 将BGR格式转换为RGB格式（因为OpenCV使用BGR而PIL使用RGB）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将NumPy数组转换为PIL Image对象
    image = Image.fromarray(rgb_image)

    if noisy_type=="gaussian":
        # 添加高斯噪声
        noisy_image = add_gaussian_noise(image)
        # noisy_image = add_blur(image)
    elif noisy_type=="occlusion":
        if num:
            noisy_image = add_occlusion(image,num_occ=num)
        else:
            # 添加矩形或其他形状进行遮挡
            noisy_image = add_occlusion(image)
    elif noisy_type=="scatter":
        # 添加随机的白色或黑色小点来模拟散点分布
        noisy_image = add_scatter(image)
    elif noisy_type=="ice":
        # 添加冰冻效果
        noisy_image = add_ice(image)
    elif noisy_type=="rain_effect":
        # 添加随机的白色条纹来实现 雨水效果
        noisy_image= add_rain_effect(image)
    elif noisy_type=="dust_effect":
        noisy_image = add_dust_effect(image)    
    elif noisy_type=="snow_effect":# 随机生成半透明的斑点来实现 雪花效果
        if light_position:
            img_array = np.asarray(image)
            height, width, _ = img_array.shape
            light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
            noisy_image = add_snow_effect(image,light_position=light_position)
        else:
            noisy_image = add_snow_effect(image)
    elif noisy_type=="overexposure":
        img_array = np.asarray(image)
        height, width, _ = img_array.shape
        light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
        noisy_image = apply_overexposure_pil(image,light_position)
    elif noisy_type=="white_balance":
        noisy_image = apply_white_balance_shift(image)
    elif noisy_type=="cracks":
        img_array = np.asarray(image)
        height, width, _ = img_array.shape
        light_position = [np.random.randint(0,width ),np.random.randint(0, height)]
        if num:
            noisy_image = add_cracks(image,light_position,num_cracks = num)
        else:
            noisy_image = add_cracks(image,light_position)
    elif noisy_type=="large_cracks":
        noisy_image = add_cracks_to_pil_image(image)
    elif noisy_type=="radiating_cracks":
        noisy_image = simulate_custom_zigzag_cracks(image)
    else :
        print("noisy_type not find!!!")
        return compressed_image

    # 将PIL Image对象转换为NumPy数组
    numpy_image = np.array(noisy_image)
    # 将RGB格式转换为BGR格式（因为PIL使用RGB而OpenCV使用BGR）
    noisy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # 将处理后的图像重新编码为 JPEG 格式
    _, img_encoded = cv2.imencode('.jpg', noisy_image)

    # 创建一个新的 CompressedImage 消息
    noisy_compressed_image = copy.deepcopy(compressed_image)
    noisy_compressed_image.data = np.array(img_encoded).tobytes()

    return noisy_compressed_image


def save_image(msg,camera_str):
    tstamp = msg.measurement_time

    temp_time = str(tstamp).split('.')
    if len(temp_time[1]) == 1:
        temp_time1_adj = temp_time[1] + '0'
    else:
        temp_time1_adj = temp_time[1]
    image_time = temp_time[0] + '_' + temp_time1_adj

    image_filename = camera_str + image_time + ".jpeg"
    f = open(out_folder + image_filename, 'w+b')
    f.write(msg.data)
    f.close()