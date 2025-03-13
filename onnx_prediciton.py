import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os 


labels_list = ['いわき', 'つくば', 'とちぎ', 'なにわ', '一宮', '三河', '三重', '上越', '下関', '世田谷', '久留米', '京都', '仙台', '伊勢志摩', '伊豆', '会津', '佐世保', '佐賀', '倉敷', '八戸', '八王子', '出雲', '函館', '前橋', '北九州', '北見', '千葉', '名古屋', '和歌山', '和泉', '品川', '四日市', '土浦', '堺', '多摩', '大分', '大宮', '大阪', '奄美', '奈良', '姫路', '宇都宮', '室蘭', '宮城', '宮崎', '富士山', '富山', '尾張小牧', '山口', '山形', '山梨', '岐阜', '岡山', '岡崎', '岩手', '島根', '川口', '川崎', '川越', '市原', '市川', '帯広', '平泉', '広島', '庄内', '弘前', '徳島', '愛媛', '成田', '所沢', '新潟', '旭川', '春日井', '春日部', '札幌', '杉並', '松戸', '松本', '板橋', '柏', '栃木', '横浜', '水戸', '江東', '沖縄', '沼津', '浜松', '湘南', '滋賀', '熊本', '熊谷', '白河', '盛岡', '相模', '知床', '石川', '神戸', '福井', '福山', '福岡', '福島', '秋田', '筑豊', '練馬', '群馬', '習志野', '船橋', '苫小牧', '葛飾', '袖ヶ浦', '諏訪', '豊橋', '豊田', '越谷', '足立', '那須', '郡山', '野田', '金沢', '釧路', '鈴鹿', '長岡', '長崎', '長野', '青森', '静岡', '飛騨', '飛鳥', '香川', '高崎', '高松', '高知', '鳥取', '鹿児島'] 


# Load the ONNX model
onnx_model_path = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/runs/dataset_0_region_update2_customized_efficentnetb0/weights/best_fp16.onnx"  # Replace with your ONNX file
session = ort.InferenceSession(onnx_model_path)

# Define image transformations (Modify based on your training preprocessing)
input_size = 128  # Change based on your model's expected input size
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])

def preprocess_image(image_path):
    """Preprocess the input image for ONNX model"""
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0).numpy().astype(np.float16)  # Convert to float16
    return image


def calculate_accuracy(gt_list, pt_list):
    """
    Calculate accuracy from ground truth and predicted labels.

    Args:
        gt_list (list): List of ground truth labels.
        pt_list (list): List of predicted labels.

    Returns:
        float: Accuracy percentage.
    """
    correct = sum(gt == pt for gt, pt in zip(gt_list, pt_list))
    total = len(gt_list)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Load and preprocess the image
image_path = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/test_real/いわき/いわき_aos_IMG1637_27_1_0_黄.jpg"  # Replace with your image path
input_tensor = preprocess_image(image_path)

# Extract filename from the path
image_filename = os.path.basename(image_path)

# Extract the first part before "_"
gt_label = image_filename.split("_")[0]

# Get input and output names of the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], {input_name: input_tensor})[0]

# Get predicted class
predicted_class = np.argmax(output)
print(f"Predicted Class: {predicted_class}")
print(f"Predicted Label:{labels_list[predicted_class]}")


image_dir = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset/dataset_0_region_update2/test_real/いわき"

gt_list = []
pt_list = []
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir,filename)
    input_tensor = preprocess_image(image_path)

    # Extract filename from the path
    image_filename = os.path.basename(image_path)

    # Extract the first part before "_"
    gt_label = image_filename.split("_")[0]

    # Get input and output names of the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    output = session.run([output_name], {input_name: input_tensor})[0]


    # Get predicted class
    predicted_class = np.argmax(output)
    # print(f"Predicted Class: {predicted_class}")
    # print(f"Predicted Label:{labels_list[predicted_class]}")

    gt_list.append(gt_label)
    pt_list.append(labels_list[predicted_class])

accuracy= calculate_accuracy(gt_list=gt_list,pt_list=pt_list)
print("Accuracy: ",accuracy)