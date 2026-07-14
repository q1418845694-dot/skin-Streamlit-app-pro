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
import matplotlib.font_manager as fm
import os
import seaborn as sns
import requests
from transformers import BertTokenizer, BertModel

# ======================== 自定义 CSS 样式 ========================
st.markdown("""
<style>
    /* 全局背景与字体 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    /* 主标题 */
    .main-title {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main-subtitle {
        text-align: center;
        color: #5a6b7c;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* 卡片容器 */
    .card {
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        transition: box-shadow 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.10);
    }
    /* 上传区域美化 */
    .upload-area {
        border: 2px dashed #ced4da;
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        background-color: #ffffff;
        transition: border-color 0.2s;
    }
    .upload-area:hover {
        border-color: #6c8cff;
    }
    /* 按钮样式 */
    .stButton>button {
        background-color: #4a6cf7;
        color: white;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #3a56d4;
        color: white;
    }
    /* 侧边栏移到底部，美化 */
    .footer-info {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        padding-top: 2rem;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    /* 结果卡片标题 */
    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #4a6cf7;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ======================== 中文字体设置 ========================
font_path = os.path.join(os.path.dirname(__file__), 'SIMSUN.TTC')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['宋体']
    st.sidebar.success(f"✅ 已加载中文字体：{font_path}")
else:
    st.sidebar.warning("⚠️ 未找到中文字体文件，图表中文可能显示为方框。")
plt.rcParams['axes.unicode_minus'] = False

# ======================== 页面配置 ========================
st.set_page_config(
    page_title="皮肤病智能识别 - 多模态融合",
    page_icon="🩺",
    layout="wide"
)

# 美化标题（去掉原生的 st.title，使用自定义样式）
st.markdown('<div class="main-title">🩺 皮肤病智能诊断系统</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">图像 + 症状描述 · 多模态精准融合</div>', unsafe_allow_html=True)

# -------------------- 模型下载配置 --------------------
MODEL_URL = "https://huggingface.co/datasets/adjuhui/skindiseaseAI/resolve/main/best_multimodal_model2.0.pth"
MODEL_PATH = "best_multimodal_model.pth"
CSV_PATH   = "Train_Ready.csv"

def download_file(url, local_filename, expected_size_mb=300):
    """从URL下载文件，若本地文件不存在、过小或损坏则重新下载"""
    if os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename) / (1024 * 1024)
        if file_size > expected_size_mb:
            try:
                torch.load(local_filename, map_location='cpu', weights_only=False)
                st.info(f"✅ 模型文件已存在且有效：{local_filename} ({file_size:.1f} MB)")
                return True
            except Exception as e:
                st.warning(f"⚠️ 本地模型文件损坏 ({file_size:.1f} MB)，将重新下载... 错误：{e}")
                os.remove(local_filename)
        else:
            st.warning(f"⚠️ 本地模型文件过小 ({file_size:.1f} MB)，将重新下载...")
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

        try:
            torch.load(local_filename, map_location='cpu', weights_only=False)
            st.success("✅ 模型下载并验证成功！")
            return True
        except Exception as e:
            st.error(f"❌ 下载的文件不是有效的 PyTorch 模型：{e}")
            os.remove(local_filename)
            return False
    except Exception as e:
        st.error(f"❌ 模型下载失败：{e}")
        return False

if not download_file(MODEL_URL, MODEL_PATH, expected_size_mb=100):
    st.stop()

# -------------------- 检查 CSV 文件 --------------------
if not os.path.exists(CSV_PATH):
    st.error(f"❌ 未找到 Train_Ready.csv 文件，请将其放置在应用目录下。")
    st.stop()

# -------------------- 定义多模态模型架构 --------------------
class SkinMultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.image_model.head = nn.Identity()
        img_dim = 768

        self.text_model = BertModel.from_pretrained('bert-base-chinese')
        text_dim = 768

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(img_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_model(images)
        if img_features.dim() == 4:
            if img_features.shape[-1] == 768:
                img_features = img_features.mean(dim=[1, 2])
            else:
                img_features = img_features.mean(dim=[2, 3])
        elif img_features.dim() == 3:
            img_features = img_features.mean(dim=1)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        fused = torch.cat((img_features, text_features), dim=1)
        return self.classifier(fused)

# -------------------- 全局缓存 --------------------
@st.cache_resource
def load_multimodal_model(model_path, num_classes, device):
    model = SkinMultiModalModel(num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-chinese')

@st.cache_data
def load_class_names_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    classes = sorted(list(df['Label'].unique()))
    return classes

# -------------------- 加载组件 --------------------
class_names = load_class_names_from_csv(CSV_PATH)

# ========== 英文缩写 → 中文疾病名称映射 ==========
LABEL_MAP = {
    "Herpes_Zoster": "带状疱疹",
    "Basal_Cell_Carcinoma": "基底细胞癌",
    "Melanoma": "黑色素瘤",
    "Melanocytic_Nevus": "色素痣",
    "Squamous_Cell_Carcinoma": "鳞状细胞癌",
    "Seborrheic_Keratosis": "脂溢性角化病",
    "Actinic_Keratosis":"日光性角化病",
    "Dermatofilbroma":"皮肤纤维瘤",
    "Vascular_Lesion":"血管性皮损",
    "Acne_Vulgaris":"痤疮",
    "Eczema":"湿疹",
    "Psoriasis":"银屑病",
    "Rosacea":"玫瑰痤疮",
    "Seborrheic_Dermatitis":"脂溢性皮炎",
    "Contact_Dermatitis":"接触性皮炎",
    "Urticaria":"荨麻疹",
    "Lichen_Planus":"扁平苔藓",
    "Vitiligo":"白癜风",
    "Alopecia_Areata":"斑秃",
    "Pityriasis_Rosea":"玫瑰糠疹",
    "Tinea_Corporis":"体廯",
    "Tinea_Pedis":"足廯",
    "Onychomycosis":"甲真菌病",
    "Herpes_Simplex":"单纯疱疹",
    "Impetigo":"脓疱疮",
    "Warts":"寻常疣",
    "Molluscum_Contagiosum":"传染性软疣",
    "Scabies":"疥疮",
    "Folliculitis":"毛囊炎",
    "Cellulitis":"蜂窝织炎",
    "Exanthems":"药疹\\病毒性皮疹",
    "Cyst":"皮肤囊肿",
    "Healthy":"健康",
    "HFMD":"手足口病",
}
def get_chinese_label(english_label):
    return LABEL_MAP.get(english_label, english_label)

tokenizer = load_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_multimodal_model(MODEL_PATH, len(class_names), device)

# 系统信息移到页面底部，避免侧边栏占用空间
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

# ======================== 主界面布局 ========================

# 使用两列布局，左列上传，右列结果
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # 上传区域卡片
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📤 上传皮肤镜图像")
        uploaded_img = st.file_uploader(
            label=" ",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="支持常见图像格式",
            label_visibility="collapsed"
        )
        if uploaded_img is not None:
            image = Image.open(uploaded_img).convert('RGB')
            st.image(image, caption="原始图像", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 症状输入卡片
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 症状描述")
        symptoms = st.text_area(
            label=" ",
            placeholder="例如：局部红斑、瘙痒、脱屑，持续两周...",
            help="详细描述可以帮助模型更准确判断",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # 结果显示区域
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 多模态预测结果")
    if uploaded_img is not None and symptoms.strip() != "":
        # 预测逻辑（无改动）
        img_tensor = img_transform(image).unsqueeze(0).to(device)
        encoded = tokenizer(
            symptoms,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(img_tensor, input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx  = top5_idx.cpu().numpy()[0]
        top5_labels = [get_chinese_label(class_names[i]) for i in top5_idx]

        # 显示Top-1结果
        st.markdown(f'<div class="result-title">🥇 融合诊断: **{top5_labels[0]}**</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-text">置信度: **{top5_prob[0]:.2%}**</div>', unsafe_allow_html=True)

        # Top-5 条形图（使用更柔和的颜色）
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("viridis", len(top5_prob))  # 更美观的配色
        y_pos = np.arange(len(top5_labels))
        ax.barh(y_pos, top5_prob, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("置信度", fontsize=9)
        ax.set_title("Top-5 多模态预测", fontsize=12)
        ax.set_xlim(0, 1)
        for i, (prob, label) in enumerate(zip(top5_prob, top5_labels)):
            ax.text(prob + 0.01, i, f"{prob:.2%}", va='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        # 展开显示Top-10
        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probabilities.cpu().numpy()[0]
            sorted_indices = np.argsort(all_prob)[::-1]
            sorted_labels = [get_chinese_label(class_names[i]) for i in sorted_indices[:10]]
            sorted_probs = all_prob[sorted_indices[:10]]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors2 = sns.color_palette("coolwarm", len(sorted_probs))
            ax2.barh(np.arange(len(sorted_labels)), sorted_probs, color=colors2)
            ax2.set_yticks(np.arange(len(sorted_labels)))
            ax2.set_yticklabels(sorted_labels, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("置信度", fontsize=9)
            ax2.set_title("Top-10 类别", fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)

    elif uploaded_img is not None and symptoms.strip() == "":
        st.warning("⚠️ 请输入症状描述以获得多模态融合结果。")
    else:
        st.info("👈 请先上传图像并填写症状描述")
    st.markdown('</div>', unsafe_allow_html=True)

# 页面底部信息
st.markdown('<div class="footer-info">', unsafe_allow_html=True)
st.markdown("""
**使用说明**  
1. 上传一张皮肤镜图像。  
2. 在文本框中输入详细的症状描述（中文）。  
3. 模型将结合图像和文本进行融合诊断，显示最可能的5种疾病及其置信度。  
4. 所有模型文件自动从 Hugging Face 下载，首次启动稍慢，后续使用缓存。
""")
st.markdown('</div>', unsafe_allow_html=True)
