import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
from PIL import Image, ExifTags
from collections import Counter
from ultralytics import YOLO

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AURA AI", page_icon="🛣️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
    .hazard-box { background-color: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-bottom: 20px;}
    .safe-box { background-color: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 15px; border-radius: 4px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# 2. LOAD THE AI BRAIN
@st.cache_resource
def load_model():
    try:
        model = YOLO('models/best.pt')
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_model()

# 3. DASHBOARD HEADER
st.title("🛣️ AURA: Smart City Infrastructure AI")
st.markdown("**Team 14D** | Automated Urban Road Assessment System")
st.divider()

# 4. SIDEBAR CONTROLS
st.sidebar.title("⚙️ System Controls")
confidence_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.25, 
                                         help="Lower this to catch faint cracks. Raise it to ignore shadows.")

st.sidebar.markdown("### 📍 Location Metadata Fallback")
st.sidebar.markdown("*If uploaded image lacks GPS metadata, the map will default to these coordinates:*")
simulated_lat = st.sidebar.number_input("Fallback Latitude", value=38.8951, format="%.4f")
simulated_lon = st.sidebar.number_input("Fallback Longitude", value=-77.0364, format="%.4f")

def get_exif_gps(image):
    """Digs into the hidden EXIF data of a JPG to extract physical GPS coordinates."""
    try:
        exif = image._getexif()
        if not exif:
            return None
        
        # PIL uses numerical tags for EXIF data. GPSInfo is tag 34853.
        for tag, value in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                # Convert Degrees/Minutes/Seconds to Decimal
                def to_decimal(val, ref):
                    decimal = float(val[0]) + float(val[1])/60.0 + float(val[2])/3600.0
                    return -decimal if ref in ['S', 'W'] else decimal
                
                lat = to_decimal(value[2], value[1])
                lon = to_decimal(value[4], value[3])
                return lat, lon
        return None
    except Exception:
        return None

# 5. MAIN WORKSPACE
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📸 1. Dashcam Feed")
    st.markdown("Upload a street-level image to scan for structural degradation.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
with col2:
    st.markdown("### 🖥️ 2. Inference Output")
    
    if uploaded_file is not None and model_loaded:
        # Load image via PIL to allow EXIF extraction before converting to OpenCV format
        pil_image = Image.open(uploaded_file)
        
        # Resize image slightly if it's a massive 4k smartphone photo to speed up inference
        img_array = np.array(pil_image)
        
        with st.spinner("Neural Network processing frame..."):
            # Run the YOLO Inference
            results = model.predict(img_array, conf=confidence_threshold)
            
            # Draw the bounding boxes on the image
            plotted_img = results[0].plot()
            
            # Convert BGR (OpenCV) back to RGB (Streamlit)
            final_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
            
            st.image(final_img, caption="Processed Dashcam Frame", use_container_width=True)
            
            boxes = results[0].boxes
            num_hazards = len(boxes)
            
            if num_hazards > 0:
                st.markdown(f'<div class="hazard-box"><strong>🚨 Scan Complete:</strong> Detected {num_hazards} structural hazard(s). Dispatching maintenance required.</div>', unsafe_allow_html=True)
                
                # --- THE NEW AI DESCRIPTION FEATURE ---
                class_dict = {0: "Longitudinal Crack", 1: "Transverse Crack", 2: "Alligator Crack", 3: "Pothole"}
                # Convert the raw tensor classes to integer IDs
                detected_classes = [int(c) for c in boxes.cls]
                # Count how many of each hazard exists
                counts = Counter(detected_classes)
                
                # Display metrics in a clean row
                m_cols = st.columns(len(counts))
                for idx, (cls_id, count) in enumerate(counts.items()):
                    m_cols[idx].metric(label=class_dict.get(cls_id, 'Unknown'), value=count)
                
                # --- THE NEW EXIF GPS FEATURE ---
                st.markdown("### 🗺️ 3. Geospatial Hazard Mapping")
                
                gps_coords = get_exif_gps(pil_image)
                
                if gps_coords:
                    final_lat, final_lon = gps_coords
                    st.success(f"📍 GPS Metadata extracted from image: Lat {final_lat:.4f}, Lon {final_lon:.4f}")
                else:
                    final_lat, final_lon = simulated_lat, simulated_lon
                    st.warning(f"⚠️ No GPS data found in image. Using simulated coordinates: Lat {final_lat:.4f}, Lon {final_lon:.4f}")

                st.caption("Automated Maintenance Dispatch Coordinate Plot:")
                
                # Render the advanced PyDeck Map
                map_data = pd.DataFrame({'lat': [final_lat], 'lon': [final_lon]})
                
                view_state = pdk.ViewState(
                    latitude=final_lat, 
                    longitude=final_lon, 
                    zoom=15, 
                    pitch=45 # Adds a cool 3D tilt
                )
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='[239, 68, 68, 200]', # Red alert color
                    get_radius=40,
                    pickable=True
                )
                
                st.pydeck_chart(pdk.Deck(map_style='dark', initial_view_state=view_state, layers=[layer]))
                
            else:
                st.markdown('<div class="safe-box"><strong>✅ Scan Complete:</strong> No hazards detected. Pavement structure optimal.</div>', unsafe_allow_html=True)
            
    elif not model_loaded:
        st.error("❌ AI Model not found. Please ensure 'best.pt' is in the same directory as this script.")
    else:
        st.info("Awaiting dashcam image upload...")
