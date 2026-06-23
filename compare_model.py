import numpy as np
import cv2
import torch
import onnxruntime as ort
import MNN
import inspireface as isf
from nets.mobilefacenet_from_pp import MobileFaceNet
from utils.utils import align_face_efficient, get_largest_face, resize_image

# 初始化 InspireFace SDK
isf.reload(model_name=None, resource_path='r18_head')
opt = isf.HF_ENABLE_FACE_RECOGNITION
session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)


# ============================================================================
# ArcFace 标准模板点 (112x112)
# ============================================================================
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041]   # 右嘴角
], dtype=np.float32)

FACE_CROP_SIZE = 112


# ============================================================================
# Umeyama 相似性变换 v3 (RMS缩放)
# ============================================================================
def umeyama_similarity_transform_v3(dest, src):
    """
    Umeyama 相似性变换 - 使用 RMS 缩放
    
    参数顺序: dest (目标点集/模板) 在前, src (源点集/检测到的关键点) 在后
    返回: 3x3 仿射变换矩阵
    """
    n = src.shape[0]

    # 计算均值
    src_mean = np.mean(src, axis=0)
    dest_mean = np.mean(dest, axis=0)

    # 去均值
    src_centered = src - src_mean
    dest_centered = dest - dest_mean

    # 计算协方差矩阵
    H = src_centered.T @ dest_centered

    # SVD 分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 处理镜像
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算缩放因子 (使用均方根)
    var_src = np.sum(src_centered ** 2) / n
    var_dest = np.sum(dest_centered ** 2) / n
    scale = np.sqrt(var_dest / var_src)

    # 计算平移向量
    t = dest_mean - scale * R @ src_mean

    # 构造 3x3 仿射变换矩阵
    transform = np.eye(3)
    transform[:2, :2] = scale * R
    transform[:2, 2] = t

    return transform


# ============================================================================
# 仿射变换 (使用 cv2.warpAffine)
# ============================================================================
def mnn_image_affine(image, transform_matrix, output_size=(112, 112)):
    """
    使用 cv2.warpAffine 进行仿射变换
    
    与 C++ ExecuteImageAffineProcessing 一致:
        - filterType: BILINEAR (cv2.INTER_LINEAR)
        - wrap: ZERO (cv2.BORDER_CONSTANT)
        - sourceFormat: BGR
        - destFormat: BGR
    """
    dst_w, dst_h = output_size
    
    # 提取 2x3 变换矩阵
    affine_matrix = transform_matrix[:2, :].astype(np.float32)
    
    # 执行仿射变换
    result = cv2.warpAffine(image, affine_matrix, (dst_w, dst_h), 
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return result


def mnn_extract_embedding(img_path):
    """
    使用 InspireFace SDK 提取特征
    """
    image_ori = cv2.imread(img_path) # BGR
    faces = get_largest_face(session, image_ori) # 找到最大人脸
    
    if faces is None:
        raise ValueError("未检测到人脸")
    
    # 直接使用 SDK 提取特征
    feature = session.face_feature_extract(image_ori, faces)
    
    return feature, None, None


def preprocess_image(image_path, image_size=112):
    """
    使用 Umeyama v3 (RMS缩放) + cv2.warpAffine 进行预处理
    输入格式: RGB (与 PT/ONNX/MNN 训练时一致)
    """
    # cv2.imread 返回 BGR
    image_bgr = cv2.imread(image_path)  # BGR
    
    # 使用 BGR 图像进行人脸检测和关键点提取
    faces = get_largest_face(session, image_bgr)
    if faces is not None:
        # 获取5个关键点
        five_points = session.get_face_five_key_points(faces)
        five_points_np = np.array(five_points, dtype=np.float32)
        
        # 使用 Umeyama v3 (RMS缩放) 计算变换矩阵
        transform = umeyama_similarity_transform_v3(REFERENCE_POINTS, five_points_np)
        
        # 对 BGR 图像执行仿射变换
        image_bgr_aligned = mnn_image_affine(image_bgr, transform, output_size=(image_size, image_size))
        
        # BGR 转换为 RGB (与 PT/ONNX/MNN 训练时一致)
        image = cv2.cvtColor(image_bgr_aligned, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.imread(image_path)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # 归一化: (img - 127.5) * 0.0078125
    img = (image.astype(np.float32) - 127.5) * 0.00784375
    
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    
    # 增加 batch 维度
    img = np.expand_dims(img, axis=0)
    
    return img


def load_pt_model(checkpoint_path, feature_dim=128):
    """加载 PyTorch 模型"""
    model = MobileFaceNet(feature_dim=feature_dim)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict_backbone']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_onnx_model(onnx_path):
    """加载 ONNX 模型"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def load_mnn_model(mnn_path):
    """加载 MNN 模型"""
    interpreter = MNN.Interpreter(mnn_path)
    session = interpreter.createSession()
    return interpreter, session


def inference_mnn(interpreter, session, img, normalize=True):
    """MNN 模型推理"""
    input_tensor = interpreter.getSessionInput(session)
    host_tensor = MNN.Tensor(img.shape, MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(host_tensor)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    shape = output_tensor.getShape()
    host_output = MNN.Tensor(shape, output_tensor.getDataType(),
                             np.zeros(shape, dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(host_output)
    output = np.array(host_output.getData(), copy=False).reshape(shape)
    if normalize:
        output = output / np.linalg.norm(output, axis=1, keepdims=True)
    return output.flatten()


def inference_pt(model, img):
    """PyTorch 模型推理"""
    with torch.no_grad():
        img_tensor = torch.from_numpy(img)
        output = model(img_tensor)
        # L2 归一化
        output = output.numpy()
        output = output / np.linalg.norm(output, axis=1, keepdims=True)
        return output.flatten()


def inference_onnx(session, img):
    """ONNX 模型推理"""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]
    # L2 归一化
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    return output.flatten()


def calculate_similarity(output1, output2):
    """计算两个输出的相似性"""
    # 余弦相似度
    cos_sim = np.dot(output1, output2)

    # L2 距离
    l2_dist = np.linalg.norm(output1 - output2)

    # L1 距离
    l1_dist = np.sum(np.abs(output1 - output2))

    # 均方误差
    mse = np.mean((output1 - output2) ** 2)

    # Pearson 相关系数
    pearson_corr = np.corrcoef(output1, output2)[0, 1]

    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'l1_distance': l1_dist,
        'mse': mse,
        'pearson_correlation': pearson_corr
    }


def print_similarity_result(model1_name, model2_name, similarity):
    """打印两个模型的相似性结果"""
    print(f"\n{'=' * 50}")
    print(f"{model1_name} vs {model2_name} 相似性对比:")
    print(f"{'=' * 50}")
    print(f"余弦相似度 (Cosine Similarity):     {similarity['cosine_similarity']:.8f}")
    print(f"L2 距离 (L2 Distance):              {similarity['l2_distance']:.8f}")
    print(f"L1 距离 (L1 Distance):              {similarity['l1_distance']:.8f}")
    print(f"均方误差 (MSE):                     {similarity['mse']:.16f}")
    print(f"Pearson 相关系数:                   {similarity['pearson_correlation']:.8f}")

    if similarity['cosine_similarity'] > 0.9999 and similarity['l2_distance'] < 1e-4:
        print("✅ 两个模型输出高度一致！")
    elif similarity['cosine_similarity'] > 0.99:
        print("⚠️ 两个模型输出相似，存在轻微差异。")
    else:
        print("❌ 两个模型输出差异较大，请检查模型转换或权重加载。")


def main():
    # 配置路径
    pt_checkpoint = 'checkpoint_29.pt'
    onnx_model = 'checkpoint_29_sim.onnx'
    mnn_model = 'r18_head.mnn'
    test_image = '017.jpg'  # 测试图像

    # 预处理图像
    print("预处理图像...")
    img = preprocess_image(test_image)
    print(f"输入形状: {img.shape}")

    # 加载 PyTorch 模型
    print(f"\n加载 PyTorch 模型: {pt_checkpoint}")
    pt_model = load_pt_model(pt_checkpoint, feature_dim=128)

    # 加载 ONNX 模型
    print(f"\n加载 ONNX 模型: {onnx_model}")
    onnx_session = load_onnx_model(onnx_model)
    input_info = onnx_session.get_inputs()[0]
    print(f"ONNX 输入名称: {input_info.name}, 形状: {input_info.shape}")

    # 加载 MNN 模型
    print(f"\n加载 MNN 模型: {mnn_model}")
    mnn_interpreter, mnn_session = load_mnn_model(mnn_model)

    # 推理
    print("\n开始推理...")
    pt_output = inference_pt(pt_model, img)
    print(f"PyTorch 输出形状: {pt_output.shape}")

    onnx_output = inference_onnx(onnx_session, img)
    print(f"ONNX 输出形状: {onnx_output.shape}")

    mnn_output = inference_mnn(mnn_interpreter, mnn_session, img, normalize=True)
    print(f"MNN (直接加载) 输出形状: {mnn_output.shape}")

    mnn_isf_output, aligned_image, cropped_resized = mnn_extract_embedding(test_image)
    mnn_isf_output = mnn_isf_output / np.linalg.norm(mnn_isf_output)
    print(f"MNN (InspireFace) 输出形状: {mnn_isf_output.shape}")

    # 两两比较
    print("\n" + "#" * 60)
    print("模型输出相似性对比 (vs MNN InspireFace):")
    print("#" * 60)

    # PyTorch vs MNN (InspireFace)
    sim_pt_isf = calculate_similarity(pt_output, mnn_isf_output)
    print_similarity_result("PyTorch", "MNN(InspireFace)", sim_pt_isf)

    # ONNX vs MNN (InspireFace)
    sim_onnx_mnn_isf = calculate_similarity(onnx_output, mnn_isf_output)
    print_similarity_result("ONNX", "MNN(InspireFace)", sim_onnx_mnn_isf)

    # MNN(直接) vs MNN(InspireFace)
    sim_mnn_vs_isf = calculate_similarity(mnn_output, mnn_isf_output)
    print_similarity_result("MNN(直接)", "MNN(InspireFace)", sim_mnn_vs_isf)

    # 前10个输出值对比
    print("\n" + "#" * 60)
    print("前 10 个输出值对比:")
    print("#" * 60)
    print(f"PyTorch:          {pt_output[:10]}")
    print(f"ONNX:             {onnx_output[:10]}")
    print(f"MNN(直接):        {mnn_output[:10]}")
    print(f"MNN(InspireFace): {mnn_isf_output[:10]}")

    # 输出特征统计信息
    print("\n" + "#" * 60)
    print("输出特征统计信息:")
    print("#" * 60)
    print(f"{'模型':<20} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print("-" * 80)
    print(f"{'PyTorch':<20} {pt_output.mean():>12.6f} {pt_output.std():>12.6f} {pt_output.min():>12.6f} {pt_output.max():>12.6f}")
    print(f"{'ONNX':<20} {onnx_output.mean():>12.6f} {onnx_output.std():>12.6f} {onnx_output.min():>12.6f} {onnx_output.max():>12.6f}")
    print(f"{'MNN(直接)':<20} {mnn_output.mean():>12.6f} {mnn_output.std():>12.6f} {mnn_output.min():>12.6f} {mnn_output.max():>12.6f}")
    print(f"{'MNN(InspireFace)':<20} {mnn_isf_output.mean():>12.6f} {mnn_isf_output.std():>12.6f} {mnn_isf_output.min():>12.6f} {mnn_isf_output.max():>12.6f}")


if __name__ == "__main__":
    main()
