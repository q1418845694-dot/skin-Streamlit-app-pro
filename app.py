import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from transformers import BertTokenizer, BertModel

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="皮肤病智能识别 - 多模态融合",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 皮肤病智能诊断系统 (图像 + 症状描述)")
st.markdown("上传皮肤镜图像并填写症状描述，模型将融合两者进行更精准的诊断。")

# -------------------- 模型下载配置 --------------------
# 多模态模型文件（请替换为你的实际链接）
MODEL_URL = "https://huggingface.co/datasets/adjuhui/skindiseaseAI/resolve/main/best_multimodal_model.pth"
MODEL_PATH = "best_multimodal_model.pth"
CSV_PATH   = "Train_Ready.csv"

def download_file(url, local_filename, expected_size_mb=100):
    """从URL下载文件，若本地文件太小则重新下载"""
    if os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename) / (1024 * 1024)
        if file_size > expected_size_mb:
            st.info(f"✅ 模型文件已存在：{local_filename} ({file_size:.1f} MB)")
            return True
        else:
            st.warning(f"⚠️ 本地模型文件过小 ({file_size:.1f} MB)，可能已损坏，将重新下载...")
            os.remove(local_filename)
    try:
        st.info("⏳ 正在下载多模态模型文件（约数百MB），请稍候...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0, text="下载中...")
        downloaded = 0
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = downloaded / total_size
                    progress_bar.progress(percent, text=f"下载中 {percent:.1%}")
        progress_bar.empty()
        st.success("✅ 模型下载完成！")
        return True
    except Exception as e:
        st.error(f"❌ 模型下载失败：{e}")
        return False

# 下载多模态模型
if not download_file(MODEL_URL, MODEL_PATH, expected_size_mb=100):
    st.stop()

# -------------------- 检查 CSV 文件 --------------------
if not os.path.exists(CSV_PATH):
    st.error(f"❌ 未找到 Train_Ready.csv 文件，请将其放置在应用目录下。")
    st.stop()

# -------------------- 定义多模态模型架构（与训练时完全一致） --------------------
class SkinMultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 图像分支：Swin Transformer
        self.image_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.image_model.head = nn.Identity()  # 去掉分类头
        img_dim = 768  # Swin tiny 输出特征维度

        # 文本分支：BERT
        self.text_model = BertModel.from_pretrained('bert-base-chinese')
        text_dim = 768

        # 融合分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(img_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        img_features = self.image_model(images)
        # 处理 Swin 可能的输出形状
        if img_features.dim() == 4:                # [B, H, W, C] 或 [B, C, H, W]
            if img_features.shape[-1] == 768:       # 若为 [B, 7, 7, 768]
                img_features = img_features.mean(dim=[1, 2])
            else:                                    # 若为 [B, 768, 7, 7]
                img_features = img_features.mean(dim=[2, 3])
        elif img_features.dim() == 3:                # [B, 49, 768]
            img_features = img_features.mean(dim=1)

        # 文本特征
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        # 融合
        fused = torch.cat((img_features, text_features), dim=1)
        return self.classifier(fused)

# -------------------- 全局缓存 --------------------
@st.cache_resource
def load_multimodal_model(model_path, num_classes, device):
    """加载多模态模型"""
    model = SkinMultiModalModel(num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    """加载 BERT 分词器"""
    return BertTokenizer.from_pretrained('bert-base-chinese')

@st.cache_data
def load_class_names_from_csv(csv_file):
    """从 CSV 读取类别名称（顺序与训练一致）"""
    df = pd.read_csv(csv_file)
    classes = sorted(list(df['Label'].unique()))
    return classes

# -------------------- 加载组件 --------------------
class_names = load_class_names_from_csv(CSV_PATH)
tokenizer = load_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_multimodal_model(MODEL_PATH, len(class_names), device)

st.sidebar.markdown("### ⚙️ 系统信息")
st.sidebar.markdown(f"类别数量: {len(class_names)}")
st.sidebar.markdown(f"运行设备: `{device}`")

# -------------------- 图像预处理 --------------------
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- 主界面 --------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_img = st.file_uploader(
        "📤 上传皮肤镜图像",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="支持常见图像格式"
    )
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="原始图像", use_column_width=True)

    # 新增：症状描述输入
    symptoms = st.text_area(
        "📝 请输入症状描述",
        placeholder="例如：局部红斑、瘙痒、脱屑，持续两周...",
        help="详细描述可以帮助模型更准确判断"
    )

if uploaded_img is not None and symptoms.strip() != "":
    with col2:
        st.subheader("🔍 多模态预测结果")

        # 图像预处理
        img_tensor = img_transform(image).unsqueeze(0).to(device)

        # 文本预处理
        encoded = tokenizer(
            symptoms,
            padding='max_length',
            truncation=True,
            max_length=64,          # 与训练时 Config.MAX_TEXT_LEN 一致
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # 推理
        with torch.no_grad():
            outputs = model(img_tensor, input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx  = top5_idx.cpu().numpy()[0]
        top5_labels = [class_names[i] for i in top5_idx]

        st.markdown(f"### 🥇 融合诊断: **{top5_labels[0]}**")
        st.markdown(f"置信度: **{top5_prob[0]:.2%}**")

        # Top-5 条形图
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("Blues_d", len(top5_prob))
        y_pos = np.arange(len(top5_labels))
        ax.barh(y_pos, top5_prob, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("置信度")
        ax.set_title("Top-5 多模态预测")
        ax.set_xlim(0, 1)
        for i, (prob, label) in enumerate(zip(top5_prob, top5_labels)):
            ax.text(prob + 0.01, i, f"{prob:.2%}", va='center')
        st.pyplot(fig)

        # 展开显示 Top-10
        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probabilities.cpu().numpy()[0]
            sorted_indices = np.argsort(all_prob)[::-1]
            sorted_labels = [class_names[i] for i in sorted_indices[:10]]
            sorted_probs = all_prob[sorted_indices[:10]]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(np.arange(len(sorted_labels)), sorted_probs, color='lightcoral')
            ax2.set_yticks(np.arange(len(sorted_labels)))
            ax2.set_yticklabels(sorted_labels, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("置信度")
            ax2.set_title("Top-10 类别")
            ax2.set_xlim(0, 1)
            st.pyplot(fig2)

elif uploaded_img is not None and symptoms.strip() == "":
    with col2:
        st.warning("⚠️ 请输入症状描述以获得多模态融合结果。")

else:
    with col2:
        st.info("👈 请先上传图像并填写症状描述")

st.markdown("---")
st.markdown("""
**使用说明**  
1. 上传一张皮肤镜图像。  
2. 在文本框中输入详细的症状描述（中文）。  
3. 模型将结合图像和文本进行融合诊断，显示最可能的5种疾病及其置信度。  
4. 所有模型文件自动从 Hugging Face 下载，首次启动稍慢，后续使用缓存。
""")
