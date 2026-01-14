import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrForObjectDetection
from torchvision.ops import nms
import tempfile
import os

# הגדרות הדף
st.set_page_config(
    page_title="מונה אנשים - People Counter",
    page_icon="👥",
    layout="wide"
)

MODEL_ID = "PekingU/rtdetr_r50vd"

# טעינת המודל (פעם אחת)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = RTDetrForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    
    # מציאת class id של person
    id2label = model.config.id2label
    person_id = [k for k, v in id2label.items() if v.lower() == "person"][0]
    
    return processor, model, person_id

processor, model, PERSON_ID = load_model()

@torch.no_grad()
def detect_people(
    image,
    threshold: float = 0.3,
    min_box_area: int = 700,
    ar_min: float = 1.05,
    ar_max: float = 6.5,
    nms_iou: float = 0.60,
    device: str | torch.device = None,
):
    """
    RT-DETR people detection with fast+strong post-processing:
    1) model inference + threshold
    2) keep only person
    3) geometry filter (area + aspect ratio)
    4) NMS dedupe

    Returns:
      dict: count, boxes, scores, labels, debug
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model.to(dev)
    model.eval()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # --- 1) Inference ---
    inputs = processor(images=image, return_tensors="pt").to(dev)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=dev)  # (h, w)
    det = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    debug = {
        "raw_total": int(det["boxes"].shape[0]),
        "raw_thresh": float(threshold),
    }

    # --- 2) Person-only ---
    person_mask = det["labels"] == PERSON_ID
    boxes = det["boxes"][person_mask]
    scores = det["scores"][person_mask]
    labels = det["labels"][person_mask]

    debug["person_after_thresh"] = int(boxes.shape[0])

    # Early exit
    if boxes.numel() == 0:
        return {
            "count": 0,
            "boxes": boxes.detach().cpu(),
            "scores": scores.detach().cpu(),
            "labels": labels.detach().cpu(),
            "debug": debug,
        }

    # --- 3) Geometry filter ---
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area = w * h
    ar = h / (w + 1e-6)

    # Only apply strict geometry to low-confidence detections
    score_gate = 0.50
    high_conf = scores >= score_gate

    geom_ok = (area >= float(min_box_area)) & ((ar >= float(ar_min)) & (ar <= float(ar_max)))

    # Keep all high-confidence detections, and only filter low-confidence ones
    keep = high_conf | geom_ok

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    debug["person_after_geom"] = int(boxes.shape[0])
    debug["geom_params"] = {
        "min_box_area": min_box_area,
        "ar_min": ar_min,
        "ar_max": ar_max,
        "score_gate": score_gate,
        "rule": "keep if high_conf OR (area>=min_area OR person-like AR)"
    }

    # --- 4) NMS dedupe ---
    keep_idx = nms(boxes, scores, float(nms_iou))
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    debug["person_after_nms"] = int(boxes.shape[0])
    debug["nms_iou"] = nms_iou

    return {
        "count": int(boxes.shape[0]),
        "boxes": boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "labels": labels.detach().cpu(),
        "debug": debug,
    }

def draw_boxes(image, detection_result):
    """ציור תיבות על התמונה"""
    img_array = np.array(image)
    boxes = detection_result["boxes"]
    scores = detection_result["scores"]
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Person {i+1}: {score:.2f}'
        cv2.putText(img_array, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_array

# כותרת
st.title("👥 מונה אנשים באמצעות RT-DETR")
st.markdown("העלה תמונה או וידאו וקבל ספירה אוטומטית של מספר האנשים")

# הגדרות מתקדמות
with st.sidebar:
    st.header("⚙️ הגדרות זיהוי")
    threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    min_box_area = st.number_input("Min Box Area (pixels)", 100, 5000, 700, 100)
    ar_min = st.slider("Min Aspect Ratio", 0.5, 3.0, 1.05, 0.05)
    ar_max = st.slider("Max Aspect Ratio", 2.0, 10.0, 6.5, 0.5)
    nms_iou = st.slider("NMS IoU Threshold", 0.3, 0.9, 0.60, 0.05)
    
    show_debug = st.checkbox("הצג מידע debug", value=False)

# בחירת סוג הקלט
input_type = st.radio("בחר סוג קלט:", ["תמונה 📷", "וידאו 🎥"])

if input_type == "תמונה 📷":
    uploaded_file = st.file_uploader("העלה תמונה", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("תמונה מקורית")
            st.image(image, use_container_width=True)
        
        with st.spinner('מזהה אנשים עם RT-DETR...'):
            result = detect_people(
                image,
                threshold=threshold,
                min_box_area=min_box_area,
                ar_min=ar_min,
                ar_max=ar_max,
                nms_iou=nms_iou
            )
            
            annotated_image = draw_boxes(image, result)
        
        with col2:
            st.subheader("תוצאה")
            st.image(annotated_image, use_container_width=True)
        
        # הצגת התוצאה
        st.success(f"🎯 נמצאו **{result['count']}** אנשים בתמונה!")
        
        # מידע debug
        if show_debug:
            with st.expander("🔍 Debug Information"):
                st.json(result['debug'])
                
                # טבלת ציונים
                if result['count'] > 0:
                    st.write("**Confidence Scores:**")
                    for i, score in enumerate(result['scores']):
                        st.write(f"Person {i+1}: {score:.3f}")

else:  # וידאו
    uploaded_video = st.file_uploader("העלה וידאו", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        process_every_n_frames = st.slider("עבד כל N פריימים", 1, 30, 5)
        
        if st.button("🎬 התחל עיבוד"):
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            col1, col2, col3 = st.columns(3)
            current_count_placeholder = col1.empty()
            max_count_placeholder = col2.empty()
            avg_count_placeholder = col3.empty()
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_people = 0
            people_per_frame = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % process_every_n_frames == 0:
                    # המרה ל-PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # זיהוי אנשים
                    result = detect_people(
                        pil_image,
                        threshold=threshold,
                        min_box_area=min_box_area,
                        ar_min=ar_min,
                        ar_max=ar_max,
                        nms_iou=nms_iou
                    )
                    
                    people_count = result['count']
                    people_per_frame.append(people_count)
                    max_people = max(max_people, people_count)
                    
                    # ציור תיבות
                    boxes = result["boxes"]
                    scores = result["scores"]
                    
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person {i+1}: {score:.2f}'
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # הצגה
                    stframe.image(frame, channels="BGR", use_container_width=True)
                    
                    # עדכון סטטיסטיקות
                    current_count_placeholder.metric("פריים נוכחי", people_count)
                    max_count_placeholder.metric("מקסימום", max_people)
                    avg_people = np.mean(people_per_frame) if people_per_frame else 0
                    avg_count_placeholder.metric("ממוצע", f"{avg_people:.1f}")
                
                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress)
            
            cap.release()
            os.unlink(tfile.name)
            
            st.success("✅ עיבוד הסתיים!")
            
            # סיכום סופי
            st.subheader("📊 סיכום")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("מקסימום אנשים", max_people)
            with col2:
                avg_people = np.mean(people_per_frame) if people_per_frame else 0
                st.metric("ממוצע אנשים", f"{avg_people:.1f}")
            with col3:
                st.metric("פריימים שנותחו", len(people_per_frame))

# מידע נוסף
with st.expander("ℹ️ מידע על האפליקציה"):
    st.markdown("""
    ### איך זה עובד?
    - האפליקציה משתמשת במודל **RT-DETR** (Real-Time Detection Transformer) מ-Peking University
    - RT-DETR הוא מודל transformer מתקדם לזיהוי אובייקטים בזמן אמת
    - הזיהוי כולל 4 שלבים:
      1. **Inference** - הרצת המודל על התמונה
      2. **Person Filter** - סינון רק זיהויים של אנשים
      3. **Geometry Filter** - סינון לפי גודל ויחס גובה-רוחב
      4. **NMS** - הסרת כפילויות
    
    ### פרמטרים:
    - **Confidence Threshold**: רמת ביטחון מינימלית לזיהוי
    - **Min Box Area**: שטח מינימלי לתיבה בפיקסלים
    - **Aspect Ratio**: יחס גובה לרוחב (אנשים בדרך כלל 1.05-6.5)
    - **NMS IoU**: threshold להסרת כפילויות
    
    ### טיפים:
    - הורד את ה-threshold אם אתה מפספס אנשים
    - הגדל את min_box_area אם יש הרבה false positives קטנים
    - התאם את aspect ratio לסוג התמונות שלך
    """)

st.markdown("---")
st.markdown("נוצר באמצעות Streamlit 🎈 ו-RT-DETR 🚀")