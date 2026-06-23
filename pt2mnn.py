from easydict import EasyDict
import yaml

from nets.arcface import *
import subprocess

def convert(cfg_path, **kwargs):

    config = yaml.load(open(cfg_path, encoding='utf8'), Loader=yaml.FullLoader)
    config = EasyDict(config)  # convert to dict
    model_config = config.Arch

    global_config = config.Global
    device = global_config.device

    model = Arcface(model_config.loss, model_config.margin_list, model_config.interclass_filtering_threshold,
                        model_config.embedding_size, 1000, model_config.sample_rate, model_config.fp16,
                        model_config.backbone, model_config.pretrained, False, device, **kwargs)

    backbone_state_dict = torch.load(model_config.pretrained, map_location='cpu')['state_dict_backbone']


    model.backbone.load_state_dict(backbone_state_dict, strict=True)
    model = model.backbone.eval()


    dummy_input = torch.randn(1, 3, 112, 112)


    onnx_file = model_config.pretrained.split('.')[0] + '.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=['input'],
        output_names=['output'],
        # opset_version=15,          # 不设置默认是20
        do_constant_folding=True   # 启用常量折叠优化
    )
    print(onnx_file + " succeed")



    import onnx
    from onnxsim import simplify

    model_onnx = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx)
    if check:
        onnx_path = f"{onnx_file.split('.')[0]}_sim.onnx"
        onnx.save(model_simp, onnx_path)
        print("简化成功")
    else:
        print("简化验证失败，保留原始模型")


    #  转mnn mnnconvert -f ONNX --modelFile model_data/mobilefacenet_128_from_pp_sim.onnx --MNNModel model_data/mobilefacenet_128_from_pp_sim.mnn --bizCode MNN
    mnn_path =  onnx_path.replace('onnx', 'mnn') 
    
    cmd = [
        'mnnconvert',
        '-f', 'ONNX',
        '--modelFile', onnx_path,
        '--MNNModel', mnn_path,
        '--bizCode', 'MNN'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("MNN 转换成功")


if __name__ == '__main__':


    cfg_path = 'config/config_convert.yaml'
    convert(cfg_path)