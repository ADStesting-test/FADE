import sys
# 将上级目录添加到 sys.path
sys.path.append("/apollo/modules/myTest")

import occlusion
import add_noise

import numpy as np
from PIL import Image, ImageDraw
import random
import sys
import os



if __name__ == "__main__":
    dir_path = '/apollo/modules/myTest/'
    image_path = dir_path+'image/origin.jpeg'

    # 打开图像
    image = Image.open(image_path)
    
    # image_c = occlusion.add_circle(image)
    # image_c.save(dir_path+'image/ircle.jpeg')

    # image_r = occlusion.add_rectangle(image)
    # image_r.save(dir_path+'image/rectangle.jpeg')

    # image_p = occlusion.add_polygon(image)
    # image_p.save(dir_path+'image/polygon.jpeg')

    # image_ir = occlusion.add_irregular_patch(image)
    # image_ir.save(dir_path+'image/irregular+.jpeg')

    # image_ir = add_noise.process_image_test(image,"occlusion",num = 30)
    # image_ir.save(dir_path+'image/irregular + litter_pitch.jpeg')

    # image_gauss = add_noise.process_image_test(image,"gaussian")
    # image_gauss.save(dir_path+'image/gaussian.jpeg')

    # image_scatter = add_noise.process_image_test(image,"scatter")
    # image_scatter.save(dir_path+'image/scatter.jpeg')

    # image_ic = add_noise.process_image_test(image,"ice")
    # image_ic.save(dir_path+'image/ice.jpeg')

    # image_rain_effect = add_noise.process_image_test(image,"rain_effect")
    # image_rain_effect.save(dir_path+'image/rain_effect.jpeg')

    image_snow_effect = add_noise.process_image_test(image,"dust_effect")
    image_snow_effect.save(dir_path+'image/dust_effect.jpeg')

    # image_snow_effect = add_noise.process_image_test(image,"snow_effect",light_position=True)
    # image_snow_effect.save(dir_path+'image/snow_effect+light.jpeg')

    # image_overexposure = add_noise.process_image_test(image,"overexposure")
    # image_overexposure.save(dir_path+'image/overexposure2.jpeg')

    # image_white_balance = add_noise.process_image_test(image,"white_balance")
    # image_white_balance.save(dir_path+'image/white_balance+1.75+1.75+0.5.jpeg')

    # image_cracks = add_noise.process_image_test(image,"cracks",num=100)
    # image_cracks.save(dir_path+'image/cracks.jpeg')

    # image_large_cracks = add_noise.process_image_test(image,"large_cracks")
    # image_large_cracks.save(dir_path+'image/large_cracks.jpeg')

    # image_large_cracks = add_noise.process_image_test(image,"radiating_cracks")
    # image_large_cracks.save(dir_path+'image/radiating_cracks.jpeg')


