import os
import random

import cv2
import numpy as np
from tqdm import tqdm

def add_lens_flare(img, flare_intensity=0.3, flare_center=None):
    """
    添加镜头光晕效果 (随机位置或指定中心)
    """
    h, w = img.shape[:2]
    if flare_center is None:
        # 随机放在图像上半部分
        flare_center = (0, 0)

    # 生成光晕蒙版 (径向渐变，中心亮)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - flare_center[0]) ** 2 + (y - flare_center[1]) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) / 2
    mask = 1 - np.clip(dist / max_dist, 0, 1)  # 中心1，边缘0
    mask = np.power(mask, 0.5)  # 衰减

    # 光晕颜色 (淡黄色)
    flare_color = np.array([1, 1, 1])  # BGR 浅橙黄
    flare_img = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        flare_img[:, :, c] = mask * flare_color[c] * flare_intensity

    img_float = img.astype(np.float32) / 255.0
    out = img_float + flare_img
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def add_white_glow(img, intensity=0.35, blur_radius=4):
    """
    在图像上添加白色光晕（整体）
    intensity: 光晕强度 (0-0.5)
    blur_radius: 高斯模糊半径（越大光晕越扩散）（3 ~ 5）
    """
    # 创建一个白色图层
    h, w = img.shape[:2]
    white_layer = np.full((h, w, 3), 255, dtype=np.uint8)

    # 对白色图层进行高斯模糊
    blurred = cv2.GaussianBlur(white_layer, (blur_radius, blur_radius), 0)

    # 将模糊后的白色图层与原图混合
    img_float = img.astype(np.float32) / 255.0
    glow_float = blurred.astype(np.float32) / 255.0
    result = img_float + glow_float * intensity
    result = np.clip(result, 0, 1)

    return (result * 255).astype(np.uint8)


def add_radial_white_glow(img, center=None, max_radius=None, intensity=0.6, falloff=2):
    """
    从中心向外添加白色径向光晕
    center: (x, y)，默认图像中心
    max_radius: 最大半径（默认图像对角线的一半）
    falloff: 衰减指数（越大光晕越集中在中心）
    """
    h, w = img.shape[:2]
    if center is None:
        center = (0, 0)
    if max_radius is None:
        max_radius = np.sqrt(w ** 2 + h ** 2) / 2

    # 生成径向距离矩阵
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    # 归一化距离 (0=中心, 1=边缘)
    dist_norm = np.clip(dist / max_radius, 0, 1)
    # 计算光晕强度：中心为1，边缘为0，指数衰减
    glow_mask = (1 - dist_norm) ** falloff

    # 白色光晕
    glow = np.stack([glow_mask, glow_mask, glow_mask], axis=-1)
    glow = glow * intensity

    img_float = img.astype(np.float32) / 255.0
    result = img_float + glow
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def add_local_white_glow(img, center=None, radius=200, intensity=0.5, blur=25):
    """
    在指定位置添加一个圆形白色光晕
    """
    h, w = img.shape[:2]
    if center is None:
        center = (0, 0)  # 默认左上角三分之一处

    # 创建白色圆盘
    white_circle = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(white_circle, center, radius, (255, 255, 255), -1)

    # 高斯模糊使边缘柔和
    blurred = cv2.GaussianBlur(white_circle, (blur, blur), 0)

    # 与原图混合
    img_float = img.astype(np.float32) / 255.0
    glow_float = blurred.astype(np.float32) / 255.0
    result = img_float + glow_float * intensity
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def backlight_with_white_glow(img, dark_factor=0.5, glow_intensity=0.1, glow_center=None):
    """
    先径向压暗中心，再在光源方向添加白色光晕
    dark_factor： 0.5 ~ 0.9
    glow_intensity：0.1 ~ 0.5
    glow_center：(0, 0) 左上  (w, 0) 右上


    """
    h, w = img.shape[:2]
    img_float = img.astype(np.float32) / 255.0

    # 1. 径向亮度衰减（中心暗，四周亮）
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radial = radius / np.max(radius)
    brightness = dark_factor + (1 - dark_factor) * radial
    img_dark = img_float * brightness[..., np.newaxis]

    # 2. 添加白色光晕（位置通常在背光来源方向，比如图像顶部或左上）
    if glow_center is None:
        # 来自左上角
        glow_center = (0, 0)

        # 来自右上角
        # glow_center = (w, 0)

    # 计算光晕强度（随距离衰减）
    dist = np.sqrt((x - glow_center[0]) ** 2 + (y - glow_center[1]) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) / 2
    glow_mask = 1 - np.clip(dist / max_dist, 0, 1)
    glow_mask = glow_mask ** 1  # 平方衰减，更集中
    glow = np.stack([glow_mask, glow_mask, glow_mask], axis=-1) * glow_intensity

    result = img_dark + glow
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or '.jpg'
    cv2.imencode(ext, img)[1].tofile(path)


def random_backlight_params(w):
    """backlight_with_white_glow 参数随机采样"""
    dark_factor = random.uniform(0.5, 0.9)
    glow_intensity = random.uniform(0.1, 0.5)
    glow_center = random.choice([(0, 0), (w, 0)])
    return dict(dark_factor=dark_factor, glow_intensity=glow_intensity, glow_center=glow_center)


def random_white_glow_params():
    """add_white_glow 参数随机采样（blur_radius 取奇数 3~5）"""
    intensity = random.uniform(0.0, 0.5)
    blur_radius = random.choice([3, 5])
    return dict(intensity=intensity, blur_radius=blur_radius)


def enhance_image(img):
    h, w = img.shape[:2]
    backlight_kw = random_backlight_params(w)
    glow_kw = random_white_glow_params()
    result = backlight_with_white_glow(img, **backlight_kw)
    result = add_white_glow(result, **glow_kw)
    params = {**backlight_kw, **glow_kw}
    params['glow_center'] = list(params['glow_center'])
    return result, params


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def iter_images(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def variant_rel_path(rel_path, variant_idx):
    """原图相对路径 -> 带变体编号的路径，如 a/b.jpg -> a/b_v2.jpg"""
    base, ext = os.path.splitext(rel_path)
    return f'{base}_v{variant_idx}{ext}'


def batch_enhance(input_dir, output_dir, variants_range=(2, 5), seed=None):
    if seed is not None:
        random.seed(seed)

    vmin, vmax = variants_range
    # if vmin < 1 or vmax < vmin:
    #     raise ValueError(f'variants_range 无效: {variants_range}')

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    image_paths = list(iter_images(input_dir))
    if not image_paths:
        print(f'未找到图像: {input_dir}')
        return

    log_path = os.path.join(output_dir, 'enhance_params.log')
    os.makedirs(output_dir, exist_ok=True)

    ok, fail, total_variants = 0, 0, 0
    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write('source_path\tvariant_count\toutput_path\tvariant\tparams\n')
        for src in tqdm(image_paths):
            rel = os.path.relpath(src, input_dir)
            img = imread_unicode(src)
            if img is None:
                fail += 1
                print(f'读取失败: {src}')
                continue

            n_variants = random.randint(vmin, vmax)
            total_variants += n_variants
            for v in range(1, n_variants + 1):
                rel_out = variant_rel_path(rel, v)
                dst = os.path.join(output_dir, rel_out)
                try:
                    out, params = enhance_image(img)
                    imwrite_unicode(dst, out)
                    log_f.write(f'{rel}\t{n_variants}\t{rel_out}\t{v}\t{params}\n')
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f'处理失败: {src} v{v} -> {e}')

    print(
        f'完成: 原图 {len(image_paths)} 张, 每张随机 {vmin}~{vmax} 种光照 '
        f'(共生成 {total_variants} 张), 成功 {ok}, 失败 {fail}, 输出目录 {output_dir}'
    )
    print(f'参数记录: {log_path}')


if __name__ == '__main__':

    VARIANTS_RANGE = (1, 1)  

    input_dir = 'distill_dataset_75_25/train'
    output_dir = 'distill_dataset_75_25/train_light'


    batch_enhance(input_dir, output_dir, variants_range=VARIANTS_RANGE, seed=None)
