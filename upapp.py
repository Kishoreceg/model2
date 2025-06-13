import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------- Page‑level settings ----------
st.set_page_config(
    page_title="Underwater Enhancer & Fish/Coral Detector",
    page_icon="🌊",
    layout="wide",                     # fills the screen
    initial_sidebar_state="expanded"   # sidebar visible on load
)

# ---------- Optional: widen / recolor sidebar (uncomment to use) ----------
# st.markdown(
#     """
#     <style>
#     section[data-testid="stSidebar"] {
#         width: 300px !important;        /* widen */
#         background-color: #0e1117 !important; /* match dark body */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# ---------- Header area with three logos ----------
col_logo1, col_title, col_logo2, col_logo3 = st.columns([1, 2, 1, 1], gap="medium")

with col_logo1:
    st.image("https://i.imgur.com/LxRVMiu.png", width=80)  # IIT Ropar logo (example)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0'>SAMUDRA</h1>"
        "<p style='margin-top:0'>Dive Deeper, See Clearer</p>",
        unsafe_allow_html=True
    )
with col_logo2:
    st.image("https://i.imgur.com/fbWa7yh.png", width=160)  # MoES logo (example)
with col_logo3:
    st.image("https://i.imgur.com/LjdGJ5d.png", width=85)   # NIOT logo (example)

st.markdown("---")

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload Image (JPG • JPEG • PNG, ≤200 MB)",              # label
    type=["jpg", "jpeg", "png"]
)

# ---------- Core image‑processing helpers ----------
def enhance_image(bgr):
    """CLAHE on L‑channel → white balance → sharpening."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    bgr_eq = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    wb = cv2.xphoto.createSimpleWB()
    bgr_wb = wb.balanceWhite(bgr_eq)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(bgr_wb, -1, kernel)

def detect_coral(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
    mask  = cv2.bitwise_or(mask1, mask2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 800]

def detect_fish(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if 500 < cv2.contourArea(c) < 50000]

def draw_boxes(bgr, fish_boxes, coral_boxes):
    out = bgr.copy()
    for x, y, w, h in fish_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for x, y, w, h in coral_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out

# ---------- Main workflow ----------
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    enhanced     = enhance_image(bgr)
    fish_boxes   = detect_fish(enhanced)
    coral_boxes  = detect_coral(enhanced)
    detections   = draw_boxes(enhanced, fish_boxes, coral_boxes)

    col1, col2, col3 = st.columns(3, gap="small")
    col1.subheader("Original")
    col1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    col2.subheader("Enhanced")
    col2.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_column_width=True)

    col3.subheader("Detected")
    col3.image(cv2.cvtColor(detections, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.success(f"**Fish detected**: {len(fish_boxes)} &nbsp;&nbsp;•&nbsp;&nbsp; **Coral regions**: {len(coral_boxes)}")
    st.info(
        "This demo uses colour/shape heuristics for speed. "
        "For production, train a small YOLO/SSD model on datasets like *Sea‑Thru* or *CoralNet*."
    )
else:
    st.sidebar.info("⬅️  Upload an image to begin")

