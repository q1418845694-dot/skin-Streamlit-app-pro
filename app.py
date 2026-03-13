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
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="皮肤病智能诊断 - 多模态", page_icon="🩺", layout="wide")
st.title("🩺 皮肤病智能诊断系统 (图像 + 症状描述)")
st.markdown("上传皮肤图像并填写症状描述，模型将融合两者进行更精准的诊断。")

# -------------------- Hugging Face 配置 --------------------
REPO_ID = "adjuhui/skindiseaseAI"          # 你的数据集仓库
FILENAME = "best_multimodal_model.pth"     # 模型文件名
CSV_PATH = "Train_Ready.csv"

@st.cache_resource
def download_model(repo_id, filename):
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=".",
            local_dir_use_symlinks=False,
            resume=True
        )
        torch.load(local_path, map_location='cpu', weights_only=False)  # 验证
        st.sidebar.success(f"✅ 模型加载成功：{filename}")
        return local_path
    except Exception as e:
        st.error(f"❌ 模型下载或验证失败：{e}")
        return None

MODEL_PATH = download_model(REPO_ID, FILENAME)
if MODEL_PATH is None:
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error("❌ 未找到 Train_Ready.csv 文件")
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

# -------------------- 加载类别和中文映射 --------------------
class_names = load_class_names_from_csv(CSV_PATH)

# 中英文映射（请根据实际类别补全）
chinese_names = {
    'Herpes_Zoster': '带状疱疹',
    'Basal_Cell_Carcinoma': '基底细胞癌',
    'Melanoma': '黑色素瘤',
    'Melanocytic_Nevus': '色素痣',
    'Squamous_Cell_Carcinoma': '鳞状细胞癌',
    'Seborrheic_Keratosis': '脂溢性角化病',
    'Actinic_Keratosis':'日光性角化病',
    'Dermatofilbroma':'皮肤纤维瘤',
    'Vascular_Lesion':'血管性皮损',
    'Acne_Vulgaris':'痤疮',
    'Eczema':'湿疹',
    'Psoriasis':'银屑病',
    'Rosacea':'玫瑰痤疮',
    'Seborrheic_Dermatitis':'脂溢性皮炎',
    'Contact_Dermatitis':'接触性皮炎',
    'Urticaria':'荨麻疹',
    'Lichen_Planus':'扁平苔藓',
    'Vitiligo':'白癜风',
    'Alopecia_Areata':'斑秃',
    'Pityriasis_Rosea':'玫瑰糠疹',
    'Tinea_Corporis':'体廯',
    'Tinea_Pedis':'足廯',
    'Onychomycosis':'甲真菌病',
    'Herpes_Simplex':'单纯疱疹',
    'Impetigo':'脓疱疮',
    'Warts':'寻常疣',
    'Molluscum_Contagiosum':'传染性软疣',
    'Scabies':'疥疮',
    'Folliculitis':'毛囊炎',
    'Cellulitis':'蜂窝织炎',
    'Exanthems':'药疹\病毒性皮疹',
    'Cyst':'皮肤囊肿',
}

def get_chinese_name(eng_name):
    return chinese_names.get(eng_name, eng_name)

# -------------------- 加载模型和分词器 --------------------
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
    uploaded_img = st.file_uploader("📤 上传皮肤镜图像", type=['jpg','jpeg','png','bmp','tiff'])
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="原始图像", use_column_width=True)

    symptoms = st.text_area("📝 请输入症状描述", placeholder="例如：局部红斑、瘙痒、脱屑...")

if uploaded_img is not None and symptoms.strip() != "":
    with col2:
        st.subheader("🔍 多模态预测结果")
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
        top5_labels = [class_names[i] for i in top5_idx]
        top5_labels_ch = [get_chinese_name(name) for name in top5_labels]

        st.markdown(f"### 🥇 融合诊断: **{top5_labels_ch[0]}**")
        st.markdown(f"置信度: **{top5_prob[0]:.2%}**")

        # Top-5 条形图
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("Blues_d", len(top5_prob))
        y_pos = np.arange(len(top5_labels_ch))
        ax.barh(y_pos, top5_prob, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_labels_ch, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("置信度")
        ax.set_title("Top-5 多模态预测")
        ax.set_xlim(0, 1)
        for i, (prob, label) in enumerate(zip(top5_prob, top5_labels_ch)):
            ax.text(prob + 0.01, i, f"{prob:.2%}", va='center')
        st.pyplot(fig)

        # Top-10 折叠
        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probabilities.cpu().numpy()[0]
            sorted_indices = np.argsort(all_prob)[::-1]
            sorted_labels_ch = [get_chinese_name(class_names[i]) for i in sorted_indices[:10]]
            sorted_probs = all_prob[sorted_indices[:10]]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(np.arange(len(sorted_labels_ch)), sorted_probs, color='lightcoral')
            ax2.set_yticks(np.arange(len(sorted_labels_ch)))
            ax2.set_yticklabels(sorted_labels_ch, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("置信度")
            ax2.set_title("Top-10 类别")
            ax2.set_xlim(0, 1)
            st.pyplot(fig2)

else:
    with col2:
        st.info("👈 请先上传图像并填写症状描述")

st.markdown("---")
st.markdown("""
**使用说明**  
1. 上传一张皮肤镜图像。  
2. 输入详细的症状描述（中文）。  
3. 模型将结合图像和文本进行融合诊断，显示最可能的5种疾病及其置信度。  
4. 类别名称已转换为中文显示，如需调整映射请修改代码中的 `chinese_names` 字典。
""")
