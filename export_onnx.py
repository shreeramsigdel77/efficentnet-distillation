import torch
import torch.nn as nn
from timm import create_model
import onnx
import numpy as np
import onnx.helper
import onnx.numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16

class CustomEfficientNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Correct attribute for timm models
        num_features = base_model.classifier.in_features
        # self.bn = nn.BatchNorm1d(num_features=base_model.fc5.in_features)  
        self.bn = nn.BatchNorm1d(num_features=num_features)  
        # self.fc5 = base_model.fc5
        self.fc = nn.Linear(num_features, base_model.classifier.out_features)  # Replacing fc5
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.base_model.extract_features(x)
        x = self.base_model.forward_features(x)  # timm uses forward_features instead of extract_features
        x = x.mean([2, 3])  # Global Average Pooling for final feature map
        x = self.bn(x)
        x = self.fc(x)
        x = self.relu(x)
        return x




class ExportONNX:
    def __init__(self, model_path: str, onnx_path: str, labels: list, device="cpu", mode="fp32"):
        """
        PyTorch の .pth モデルを ONNX にエクスポートするクラス。

        Args:
            model_path (str): 保存済みの PyTorch モデルのパス (例: "best.pth")
            onnx_path (str): エクスポートする ONNX モデルのパス (例: "best.onnx")
            labels_path (list): list のパス
            device (str, optional): "cuda" or "cpu" (デフォルト: "cpu")
            mode (str, optional): "fp32", "fp16", "int16" のいずれか（デフォルト: "fp32"）
        """
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.mode = mode.lower()
        self.device = torch.device(device)

        # ラベルを読み込む
        # self.labels = self._load_labels(labels_path)
        self.labels = labels
        self.num_classes = len(self.labels)  # クラス数を自動計算

        # モデルをエクスポート
        self.export()

    # def _load_labels(self, labels_path: str):
    #     """labels.txt を読み込んでリストとして返す。"""
    #     with open(labels_path, "r", encoding="utf-8") as f:
    #         labels = [line.strip() for line in f.readlines()]
    #     return labels

    def export(self):
        """PyTorch モデルを ONNX にエクスポートする。"""
        print(f"ONNX 変換中: {self.model_path} → {self.onnx_path} ({self.mode.upper()})")
        
        # `create_model()` を使用して学習時と同じモデルを作成
        init_model = create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=self.num_classes,
            drop_rate=0.3
        ).to(self.device)

        model = CustomEfficientNet(init_model)

        # `state_dict` をロード
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()  # 推論モードへ

        # モデルの変換設定
        if self.mode == "fp16":
            model.half()  # FP16 に変換

        # ダミー入力（ONNX の入出力を定義するために必要）
        dummy_input = torch.randn(1, 3, 128, 128).to(self.device)
        if self.mode == "fp16":
            dummy_input = dummy_input.half()  # FP16 に変換

        # ONNX ファイル名を決定
        onnx_output_path = self.onnx_path.replace(".onnx", f"_{self.mode}.onnx")

        # ONNX にエクスポート（FP32 または FP16）
        torch.onnx.export(
            model, dummy_input, onnx_output_path,
            export_params=True,  # モデルの重みを含める
            opset_version=11,  # ONNX のバージョン（変更可能）
            do_constant_folding=True,  # 最適化の適用
            input_names=["input"],  # 入力名
            output_names=["output"],  # 出力名
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # バッチサイズを可変に
        )

        print(f"ONNX {self.mode.upper()} モデルを {onnx_output_path} にエクスポートしました。")

        # FP16 変換（ONNX）
        if self.mode == "fp16":
            model_fp32 = onnx.load(onnx_output_path)
            model_fp16 = convert_float_to_float16(model_fp32)  # FP16 変換
            onnx.save(model_fp16, onnx_output_path)
            print(f"ONNX FP16 モデルを {onnx_output_path} にエクスポートしました。")

        # INT16 変換（ONNX）
        if self.mode == "int16":
            self.convert_onnx_to_int16(onnx_output_path)

    def convert_onnx_to_int16(self, onnx_fp32_path):
        """ONNX モデルの重みを INT16 に変換する"""
        print(f"ONNX INT16 変換中: {onnx_fp32_path}")

        model = onnx.load(onnx_fp32_path)
        
        # 各ノードのデータ型を INT16 に変換
        for tensor in model.graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                data = onnx.numpy_helper.to_array(tensor)
                scale = np.max(np.abs(data)) / (2**15)  # INT16 のスケーリング係数
                data_int16 = (data / scale).astype(np.int16)  # INT16 に変換

                # 変換した INT16 のデータをセット
                tensor.CopyFrom(onnx.numpy_helper.from_array(data_int16, name=tensor.name))

        # INT16 版 ONNX ファイルを保存
        onnx_int16_path = onnx_fp32_path.replace("_fp32.onnx", "_int16.onnx")
        onnx.save(model, onnx_int16_path)

        print(f"ONNX INT16 モデルを {onnx_int16_path} にエクスポートしました。")



labels_list = ['いわき', 'つくば', 'とちぎ', 'なにわ', '一宮', '三河', '三重', '上越', '下関', '世田谷', '久留米', '京都', '仙台', '伊勢志摩', '伊豆', '会津', '佐世保', '佐賀', '倉敷', '八戸', '八王子', '出雲', '函館', '前橋', '北九州', '北見', '千葉', '名古屋', '和歌山', '和泉', '品川', '四日市', '土浦', '堺', '多摩', '大分', '大宮', '大阪', '奄美', '奈良', '姫路', '宇都宮', '室蘭', '宮城', '宮崎', '富士山', '富山', '尾張小牧', '山口', '山形', '山梨', '岐阜', '岡山', '岡崎', '岩手', '島根', '川口', '川崎', '川越', '市原', '市川', '帯広', '平泉', '広島', '庄内', '弘前', '徳島', '愛媛', '成田', '所沢', '新潟', '旭川', '春日井', '春日部', '札幌', '杉並', '松戸', '松本', '板橋', '柏', '栃木', '横浜', '水戸', '江東', '沖縄', '沼津', '浜松', '湘南', '滋賀', '熊本', '熊谷', '白河', '盛岡', '相模', '知床', '石川', '神戸', '福井', '福山', '福岡', '福島', '秋田', '筑豊', '練馬', '群馬', '習志野', '船橋', '苫小牧', '葛飾', '袖ヶ浦', '諏訪', '豊橋', '豊田', '越谷', '足立', '那須', '郡山', '野田', '金沢', '釧路', '鈴鹿', '長岡', '長崎', '長野', '青森', '静岡', '飛騨', '飛鳥', '香川', '高崎', '高松', '高知', '鳥取', '鹿児島'] 

# 実行例
exporter = ExportONNX(
    model_path="./runs/dataset_0_region_update2_customized_efficentnetb0/weights/best.pth",
    onnx_path="./runs/dataset_0_region_update2_customized_efficentnetb0/weights/best.onnx",
    labels=labels_list,
    mode="fp16"
)



