import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Analytics | AI DAM", layout="wide")

# ---------- UNIFIED UI CSS ----------
st.markdown("""
<style>
    /* GLOBAL THEME SYNC */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: #e2e8f0;
    }

    /* GLASSMORPHISM CARDS */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    .stat-val { 
        font-size: 32px; 
        font-weight: 800; 
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    
    .stat-label { 
        font-size: 14px; 
        color: #94a3b8; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .note-box {
        background: rgba(56, 189, 248, 0.05);
        border-left: 4px solid #38bdf8;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* TABLE STYLING */
    .stTable {
        background-color: transparent;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("📊 System Analytics")
st.markdown("Deep dive into the vector space and dataset health.")

# ---------- DATA LOADING ----------
@st.cache_resource
def load_data():
    try:
        # Use relative pathing from project root
        base_path = os.getcwd()
        pickle_path = os.path.join(base_path, "embeddings", "image_embeddings.pkl")
        
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, tuple) and len(data) > 0:
            actual_dict = data[0]
            if isinstance(actual_dict, dict):
                return actual_dict
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {}

image_embeddings = load_data()
doc_files = os.listdir("DataSet/Documents") if os.path.exists("DataSet/Documents") else []

# ---------- TOP LEVEL METRICS ----------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f'<div class="metric-card"><div class="stat-label">Total Images</div><div class="stat-val">{len(image_embeddings)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="stat-label">Total Documents</div><div class="stat-val">{len(doc_files)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="stat-label">Vector Dims</div><div class="stat-val">512</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="stat-label">Model Engine</div><div class="stat-val">CLIP ViT-B/32</div></div>', unsafe_allow_html=True)

st.divider()

# ---------- TWO COLUMN LAYOUT ----------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("💡 Embedding Insights")
    
    if image_embeddings:
        all_embeds = np.array(list(image_embeddings.values()))
        mean_val, std_val = np.mean(all_embeds), np.std(all_embeds)
        
        # UI IMPROVEMENT: Using a stylized container for stats
        with st.container(border=True):
            stats_df = pd.DataFrame({
                "Metric": ["Global Mean", "Std Deviation", "Min Value", "Max Value"],
                "Value": [f"{mean_val:.4f}", f"{std_val:.4f}", f"{np.min(all_embeds):.4f}", f"{np.max(all_embeds):.4f}"]
            })
            st.table(stats_df)
        
        st.markdown("""
        <div class="note-box">
            <b>Pro Tip:</b> CLIP vectors are high-dimensional. A <b>Standard Deviation</b> that is consistent across your dataset suggests your images are well-distributed, making natural language search more accurate.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No image embeddings found to analyze.")

with right_col:
    st.subheader("🎯 Internal Similarity Distribution")
    
    if len(image_embeddings) > 1:
        # Sample similarities
        embed_list = list(image_embeddings.values())[:50] 
        scores = []
        
        for i in range(len(embed_list)):
            for j in range(i + 1, len(embed_list)):
                v1, v2 = embed_list[i].flatten(), embed_list[j].flatten()
                v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
                scores.append(np.dot(v1, v2))
        
        hist_values, bin_edges = np.histogram(scores, bins=20)
        chart_data = pd.DataFrame({
            'Similarity': bin_edges[:-1],
            'Frequency': hist_values
        }).set_index('Similarity')
        
        # UI IMPROVEMENT: Area chart with custom color
        st.area_chart(chart_data, color="#38bdf8")
        st.caption("Distribution of image variety (0.0 = unique, 1.0 = duplicate)")
        
        with st.expander("📚 How to read this?"):
            st.write("""
            This graph shows the **Cosine Similarity** between random pairs of images in your dataset. 
            
            * **The "Sweet Spot" (0.4 - 0.6):** Most diverse datasets peak here. It means the model sees enough shared context to categorize them, but enough difference to tell them apart.
            * **The "Red Zone" (>0.85):** High spikes here usually indicate **near-duplicates**. If you see this, check if you have multiple copies of the same image or very similar burst-fire photos.
            * **The "Cold Zone" (<0.3):** Points here represent images that have almost zero conceptual overlap (e.g., a photo of a "Microchip" vs. a "Banana"). A peak here means your dataset is extremely varied.
            * **Width of the Curve:** A **wide, flat curve** indicates a balanced, healthy dataset. A **very narrow spike** means your dataset is "over-specialized" (e.g., 8,000 photos of only different types of grass).
            """)
    else:
        st.warning("Add more images to generate a similarity map.")
