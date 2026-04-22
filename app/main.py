from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ''}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.acne_features import extract_features
from app.face_detect import FaceDetectionError, detect_main_face
from app.preprocess import preprocess_face
from app.severity import calculate_severity
from app.utils import (
    ImageValidationError,
    bgr_to_rgb,
    decode_uploaded_image,
    draw_bbox,
    overlay_mask,
    save_image,
    validate_image_quality,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def analyze_image(image_bgr):
    validate_image_quality(image_bgr)
    face_result = detect_main_face(image_bgr)
    preprocessed = preprocess_face(face_result.cropped_face)
    acne_result = extract_features(preprocessed)
    severity = calculate_severity(acne_result.features)

    overlay = overlay_mask(preprocessed.bgr, acne_result.red_mask)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_image(OUTPUT_DIR / "original_with_bbox.png", draw_bbox(image_bgr, face_result.bbox))
    save_image(OUTPUT_DIR / "face_crop.png", preprocessed.display_bgr)
    save_image(OUTPUT_DIR / "red_overlay.png", overlay)

    return {
        "face_result": face_result,
        "preprocessed": preprocessed,
        "acne_result": acne_result,
        "severity": severity,
        "overlay": overlay,
    }


def render_sidebar():
    st.sidebar.header("Instructions")
    st.sidebar.write(
        "Upload a clear JPG or PNG photo with one visible face. "
        "Frontal or slightly turned faces work best."
    )
    st.sidebar.write(
        "This is a rule-based academic prototype, not a medical diagnosis tool."
    )


def render_results(results, original_image_bgr):
    face_result = results["face_result"]
    preprocessed = results["preprocessed"]
    acne_result = results["acne_result"]
    severity_result = results["severity"]
    overlay_bgr = results["overlay"]

    original_rgb = bgr_to_rgb(original_image_bgr)
    cropped_rgb = bgr_to_rgb(preprocessed.display_bgr)
    overlay_rgb = bgr_to_rgb(overlay_bgr)

    def fmt_features(features_dict):
        formatted = {}
        for k, v in features_dict.items():
            if isinstance(v, float):
                formatted[k] = round(v, 4)
            else:
                formatted[k] = v
        return formatted
    
    st.json(fmt_features(acne_result.features))

    st.subheader("Cropped Face")
    st.image(cropped_rgb, caption="Cropped Face", width=320)

    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.image(original_rgb, caption="Original Image", width=320)
    with c2:
        st.image(overlay_rgb, caption="Detected Acne-like Regions", width=320)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Severity Score", f"{severity_result.score:.2f}/100")
    with m2:
        st.metric("Severity Label", severity_result.label)

    st.write("**Explanation:**", severity_result.explanation)

    st.subheader("Extracted Features")
    st.json(fmt_features(acne_result.features))


def main():
    st.set_page_config(page_title="Acne Detector", layout="wide")
    st.title("Acne Detection & Skin Severity Analyzer")
    render_sidebar()

    uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.caption("Upload one clear image to analyze.")
        return

    image_bgr = decode_uploaded_image(uploaded.getvalue())
    st.image(bgr_to_rgb(image_bgr), caption="Original uploaded image", use_container_width=True)

    if st.button("Analyze"):
        try:
            results = analyze_image(image_bgr)
            render_results(results, image_bgr)
        except (ImageValidationError, FaceDetectionError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()