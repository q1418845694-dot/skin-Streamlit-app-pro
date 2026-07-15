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
import seaborn as sns
import os
import re
from transformers import BertConfig, BertModel, BertTokenizer

# ======================== 强制离线 ========================
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ======================== 自定义 CSS（保留您之前的样式） ========================
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', 'Roboto', sans-serif; }
    .main-title { text-align: center; padding: 1.5rem 0 0.5rem 0; color: #2c3e50; font-size: 3rem; font-weight: 700; letter-spacing: 1px; }
    .main-subtitle { text-align: center; color: #5a6b7c; font-size: 1.2rem; margin-bottom: 2rem; }
    .card { background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); padding: 1.5rem 2rem; margin-bottom: 2rem; transition: box-shadow 0.2s ease; }
    .card:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.10); }
    .result-title { font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem; }
    .confidence-text { font-size: 1.2rem; color: #4a6cf7; font-weight: 500; }
    .footer-info { text-align: center; color: #7f8c8d; font-size: 0.9rem; padding-top: 2rem; border-top: 1px solid #e9ecef; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ======================== 中文字体 ========================
font_path = os.path.join(os.path.dirname(__file__), 'SIMSUN.TTC')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['宋体']
else:
    st.sidebar.warning("⚠️ 未找到 SIMSUN.TTC，图表中文可能显示方框。")
plt.rcParams['axes.unicode_minus'] = False

# ======================== 标题 ========================
st.markdown('<div class="main-title">🩺 皮肤病智能诊断系统</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">图像 + 症状描述 · 多模态精准融合</div>', unsafe_allow_html=True)

# ======================== 本地文件配置 ========================
MODEL_PATH = "best_multimodal_model2.0.pth"
CSV_PATH   = "Train_Ready.csv"

# 检查文件
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 找不到模型文件：{MODEL_PATH}")
    st.stop()
if not os.path.exists(CSV_PATH):
    st.error(f"❌ 找不到 CSV 文件：{CSV_PATH}")
    st.stop()
file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
if file_size < 100:
    st.error(f"❌ 模型文件过小 ({file_size:.1f} MB)")
    st.stop()
try:
    torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
except Exception as e:
    st.error(f"❌ 模型损坏：{e}")
    st.stop()
st.success(f"✅ 模型文件有效 ({file_size:.1f} MB)")

# 加载类别
df = pd.read_csv(CSV_PATH, encoding='utf-8')
class_names = sorted(list(df['Label'].unique()))
num_classes = len(class_names)
st.sidebar.markdown(f"**类别数量：** {num_classes}")

# ======================== 英文→中文映射 ========================
LABEL_MAP = {
    "Herpes_Zoster": "带状疱疹", "Basal_Cell_Carcinoma": "基底细胞癌",
    "Melanoma": "黑色素瘤", "Melanocytic_Nevus": "色素痣",
    "Squamous_Cell_Carcinoma": "鳞状细胞癌", "Seborrheic_Keratosis": "脂溢性角化病",
    "Actinic_Keratosis": "日光性角化病", "Dermatofilbroma": "皮肤纤维瘤",
    "Vascular_Lesion": "血管性皮损", "Acne_Vulgaris": "痤疮",
    "Eczema": "湿疹", "Psoriasis": "银屑病", "Rosacea": "玫瑰痤疮",
    "Seborrheic_Dermatitis": "脂溢性皮炎", "Contact_Dermatitis": "接触性皮炎",
    "Urticaria": "荨麻疹", "Lichen_Planus": "扁平苔藓", "Vitiligo": "白癜风",
    "Alopecia_Areata": "斑秃", "Pityriasis_Rosea": "玫瑰糠疹",
    "Tinea_Corporis": "体廯", "Tinea_Pedis": "足廯", "Onychomycosis": "甲真菌病",
    "Herpes_Simplex": "单纯疱疹", "Impetigo": "脓疱疮", "Warts": "寻常疣",
    "Molluscum_Contagiosum": "传染性软疣", "Scabies": "疥疮",
    "Folliculitis": "毛囊炎", "Cellulitis": "蜂窝织炎",
    "Exanthems": "药疹/病毒性皮疹", "Cyst": "皮肤囊肿",
    "Healthy": "健康", "HFMD": "手足口病",
}
def get_chinese_label(eng):
    return LABEL_MAP.get(eng, eng)

# ======================== BERT 配置（手动定义，无需外部文件） ========================
def create_bert_model():
    config = BertConfig(
        vocab_size=21128, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=512, type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0
    )
    return BertModel(config)

# ======================== 多模态模型（分类头动态构建） ========================
class SkinMultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.image_model.head = nn.Identity()
        self.text_model = create_bert_model()
        self.classifier = None  # 动态构建

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

# ======================== 动态构建分类头 ========================
def build_classifier_from_state_dict(state_dict, num_classes):
    # 找出所有 classifier 的权重键（支持 classifier.1.weight 或 classifier1.weight 等）
    classifier_keys = [k for k in state_dict.keys() if k.startswith('classifier') and k.endswith('.weight')]
    if not classifier_keys:
        raise RuntimeError("未找到任何 classifier 权重键")
    
    def extract_number(key):
        # 匹配 classifier[.数字].weight 或 classifier数字.weight
        m = re.search(r'classifier[\.]?(\d+)\.weight', key)
        if m:
            return int(m.group(1))
        else:
            return float('inf')
    
    classifier_keys.sort(key=extract_number)
    
    layers = []
    prev_out = None
    for key in classifier_keys:
        weight = state_dict[key]
        bias_key = key.replace('.weight', '.bias')
        bias = state_dict.get(bias_key, None)
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        
        if prev_out is None:
            lin = nn.Linear(in_features, out_features)
        else:
            # 如果输入维度不匹配，插入适配层（通常不会发生）
            if in_features != prev_out:
                lin = nn.Linear(prev_out, out_features)
            else:
                lin = nn.Linear(in_features, out_features)
        lin.weight.data = weight
        if bias is not None:
            lin.bias.data = bias
        else:
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        prev_out = out_features

    # 如果最后一层输出不等于 num_classes，添加适配层
    if prev_out != num_classes:
        layers.append(nn.Linear(prev_out, num_classes))
    
    # 在线性层之间插入 ReLU（最后一个之后不加）
    if len(layers) > 1:
        new_layers = []
        for i, layer in enumerate(layers):
            new_layers.append(layer)
            if i != len(layers) - 1:
                new_layers.append(nn.ReLU())
        return nn.Sequential(*new_layers)
    else:
        return nn.Sequential(*layers)

# ======================== 加载模型 ========================
@st.cache_resource
def load_multimodal_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinMultiModalModel(num_classes)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # 分离 classifier 和其余部分
    classifier_state = {k: v for k, v in state_dict.items() if k.startswith('classifier')}
    other_state = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    
    # 加载非 classifier 部分（图像和文本分支）
    model.load_state_dict(other_state, strict=False)
    
    # 构建分类头
    if classifier_state:
        model.classifier = build_classifier_from_state_dict(classifier_state, num_classes)
    else:
        raise RuntimeError("权重中不包含分类头参数")
    
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_multimodal_model()
st.sidebar.markdown(f"**运行设备：** `{device}`")

# ======================== Tokenizer ========================
@st.cache_resource
def load_tokenizer():
    try:
        return BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
    except Exception:
        st.error("""
        ❌ 无法加载 BERT 分词器，因为本地缺少缓存。
        请在有网环境执行一次以下命令：
        `python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-chinese')"`
        然后重新运行本程序。
        """)
        st.stop()

tokenizer = load_tokenizer()

# ======================== 图像预处理 ========================
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================== 主界面 ========================
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📤 上传皮肤镜图像")
        uploaded_img = st.file_uploader(" ", type=['jpg','jpeg','png','bmp','tiff'], label_visibility="collapsed")
        if uploaded_img is not None:
            image = Image.open(uploaded_img).convert('RGB')
            st.image(image, caption="原始图像", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 症状描述")
        symptoms = st.text_area(" ", placeholder="例如：局部红斑、瘙痒、脱屑，持续两周...", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 多模态预测结果")
    if uploaded_img is not None and symptoms.strip() != "":
        img_tensor = img_transform(image).unsqueeze(0).to(device)
        encoded = tokenizer(symptoms, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(img_tensor, input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5)
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx = top5_idx.cpu().numpy()[0]
        top5_labels = [get_chinese_label(class_names[i]) for i in top5_idx]

        st.markdown(f'<div class="result-title">🥇 融合诊断: **{top5_labels[0]}**</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-text">置信度: **{top5_prob[0]:.2%}**</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("viridis", len(top5_prob))
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

        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probs.cpu().numpy()[0]
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
        st.warning("⚠️ 请输入症状描述")
    else:
        st.info("👈 请先上传图像并填写症状描述")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-info">', unsafe_allow_html=True)
st.markdown("""
**使用说明**  
1. 上传图像。  
2. 输入症状描述。  
3. 模型融合诊断，显示 Top-5。  
4. 所有文件从本地加载，无需联网。
""")
st.markdown('</div>', unsafe_allow_html=True)
