import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 自作したResNetクラスを別ファイルからインポート
from ResNet import ResNet

def load_and_preprocess_image(image_path, size=(224, 224)):
    # PILを使って画像をRGBモードで読み込み
    img = Image.open(image_path).convert('RGB')
    # 指定サイズにリサイズ
    img = img.resize(size)
    # PIL画像からnumpy配列を経てテンソルに変換、かつ [0, 255] から [0.0, 1.0] へ正規化
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0], 3)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0 # (H, W, C) -> (C, H, W)
    #バッチサイズ用の次元を追加 (C, H, W) -> (1, C, H, W)
    return img_tensor.unsqueeze(0)

def main():
    resnet18_arch = ((2, 64), (2, 128), (2, 256), (2, 512))
    model = ResNet(arch=resnet18_arch)

    url="https://download.pytorch.org/models/resnet18-f37072fd.pth"
    pretrained_dict=torch.hub.load_state_dict_from_url(url,progress=True)
    custom_dict={}
    custom_dict['net.b1.0.weight']=pretrained_dict['conv1.weight']
    custom_dict['net.b1.1.weight']=pretrained_dict['bn1.weight']
    custom_dict['net.b1.1.bias']=pretrained_dict['bn1.bias']
    custom_dict['net.b1.1.running_mean']=pretrained_dict['bn1.running_mean']
    custom_dict['net.b1.1.running_var']=pretrained_dict['bn1.running_var']

    stage_mapping = {2: 'layer1', 3: 'layer2', 4: 'layer3', 5: 'layer4'}
    for b_idx, layer_name in stage_mapping.items():
        for res_idx in range(2):
            for key in pretrained_dict.keys():
                prefix_official = f"{layer_name}.{res_idx}."
                if key.startswith(prefix_official):
                    sub_key = key[len(prefix_official):]
                    new_key = f"net.b{b_idx}.{res_idx}.{sub_key}"
                    custom_dict[new_key] = pretrained_dict[key]
    
    dummy_input = torch.randn(1, 3, 448, 448)
    _ = model(dummy_input)

    model.load_state_dict(custom_dict, strict=False)
    model.eval() # 推論モードに設定 (BatchNormなどを固定)

    img_size = (448, 448)
    input_A = load_and_preprocess_image('machigai02_1.png', size=img_size)
    input_B = load_and_preprocess_image('machigai02_2.png', size=img_size)

    with torch.no_grad():
        X_A,X_B=input_A,input_B
        for name,layer in model.net.named_children():
            X_A=layer(X_A)
            X_B=layer(X_B)
            if name=='b3':#b4からb3に変更
                feat_A=X_A
                feat_B=X_B
                break

    #diff_map = torch.mean((feat_A - feat_B) ** 2, dim=1,keepdim=True)
    diff_map = torch.max(torch.abs(feat_A - feat_B), dim=1, keepdim=True)[0]#絶対値の差の最大値を採用

    diff_map = F.avg_pool2d(diff_map, kernel_size=5, stride=1, padding=2)#カーネルサイズの拡大

    # 元の画像サイズ（448x448）に引き伸ばし
    diff_map_resized = F.interpolate(diff_map, size=img_size, mode='bilinear', align_corners=False)
    diff_map_numpy = diff_map_resized.squeeze().numpy()

    diff_map_numpy = (diff_map_numpy - diff_map_numpy.min()) / (diff_map_numpy.max() - diff_map_numpy.min() + 1e-8)
    # 閾値の調整
    #threshold = diff_map_numpy.max() * 0.50
    #mask = diff_map_numpy > threshold

    # 可視化処理
    original_img = Image.open('machigai02_1.png').resize(img_size)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(original_img)

    ax.imshow(diff_map_numpy, cmap='Reds', alpha=0.35)
    threshold = 0.65
    # グリッドによる枠線の描画判定（細かく見つけるためサイズを16から8に変更）
    step = 24
    for y in range(step, img_size[1] - step, step):
        for x in range(step, img_size[0] - step, step):
            local_area = diff_map_numpy[y-step:y+step, x-step:x+step]
            # 周辺で自分が一番強く、かつ閾値を超えている場合のみ、そこに間違いがあると判定
            if diff_map_numpy[y, x] == local_area.max() and diff_map_numpy[y, x] > threshold:
                # 検出したポイントに綺麗な赤丸を描く
                circle = patches.Circle(
                    (x, y), radius=20, 
                    linewidth=2.5, edgecolor='red', facecolor='none'
                )
                ax.add_patch(circle)

    plt.title("Detected Differences (High Precision)")
    plt.axis('off')
    plt.savefig('result_machigai.png', bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    main()