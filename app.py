import streamlit as st
import torch
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

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="皮肤病智能识别 - Swin Transformer",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 皮肤病智能识别系统 (Swin Transformer)")
st.markdown("上传皮肤镜图像，模型将预测其所属的病变类别。")

# -------------------- 模型下载配置 --------------------
MODEL_URL = "https://huggingface.co/datasets/adjuhui/skindiseaseAI/resolve/main/best_model(1).pth"
MODEL_PATH = "best_model(1).pth"
CSV_PATH   = "Train_Ready.csv"

def download_file(url, local_filename):
    """从URL下载文件，并显示进度条"""
    if os.path.exists(local_filename):
        st.info(f"✅ 模型文件已存在：{local_filename}")
        return True
    try:
        st.info("⏳ 正在下载模型文件（约105MB），请稍候...")
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

# 下载模型（如果本地不存在）
if not download_file(MODEL_URL, MODEL_PATH):
    st.stop()

# -------------------- 检查 CSV 文件 --------------------
if not os.path.exists(CSV_PATH):
    st.error(f"❌ 未找到 Train_Ready.csv 文件，请将其放置在应用目录下。")
    st.stop()

# -------------------- 全局缓存 --------------------
@st.cache_resource
def load_model(model_path, num_classes, device):
    """加载 Swin Transformer 模型"""
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    # 关键：weights_only=False 兼容旧版模型
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_data
def load_class_names_from_csv(csv_file):
    """从训练时使用的 CSV 文件中提取类别名称（与训练时顺序一致）"""
    df = pd.read_csv(csv_file)
    classes = sorted(list(df['Label'].unique()))   # 训练时也是 sorted
    return classes

# -------------------- 加载类别 --------------------
class_names = load_class_names_from_csv(CSV_PATH)

# -------------------- 加载模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, len(class_names), device)

st.sidebar.markdown("### ⚙️ 系统信息")
st.sidebar.markdown(f"类别数量: {len(class_names)}")
st.sidebar.markdown(f"运行设备: `{device}`")

# -------------------- 图像预处理 --------------------
val_transform = transforms.Compose([
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

if uploaded_img is not None:
    with col2:
        st.subheader("🔍 预测结果")

        input_tensor = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx  = top5_idx.cpu().numpy()[0]
        top5_labels = [class_names[i] for i in top5_idx]

        st.markdown(f"### 🥇 预测: **{top5_labels[0]}**")
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
        ax.set_title("Top-5 预测")
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

else:
    with col2:
        st.info("👈 请先上传一张皮肤图像")

st.markdown("---")
st.markdown("""
**使用说明**  
请上传您的发病部位的清晰图片，系统将为您诊断出最可能的皮肤病类型  
""")
