import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import pytesseract
import logging
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.join(os.getcwd(), 'tessdata')

import matplotlib.pyplot as plt
# or if you don't have matplotlib, we'll use PIL's show()

def auto_rotate_image(image_bytes):
    """Automatically detect and correct image rotation"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Open image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            angles = []
            for line in lines:
                theta = line[0][1]
                # Convert to degrees
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            if angles:
                # Find median angle
                median_angle = np.median(angles)
                
                # Only rotate if significant
                if abs(median_angle) > 1:
                    print(f"🔄 Rotating by {median_angle:.1f} degrees")
                    # Rotate image
                    (h, w) = open_cv_image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(open_cv_image, M, (w, h), 
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
                    
                    # Convert back to PIL
                    result = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
                    
                    output = io.BytesIO()
                    result.save(output, format='PNG')
                    output.seek(0)
                    return output.getvalue()
        
        # If no lines detected or no rotation needed
        return image_bytes
        
    except Exception as e:
        print(f"Auto-rotate error: {e}")
        return image_bytes

def save_preprocessed_image(image_bytes, output_path="preprocessed_output.png"):
    """Save preprocessed image to disk for inspection"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        img.save(output_path)
        print(f"✅ Preprocessed image saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving preprocessed image: {e}")
        return None

def show_preprocessed_image(image_bytes):
    """Display preprocessed image (if running in GUI environment)"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        img.show()  # Opens with default image viewer
        return img
    except Exception as e:
        print(f"❌ Error displaying image: {e}")
        return None

def compare_original_vs_preprocessed(image_bytes):
    """Create a side-by-side comparison"""
    try:
        # Original
        original = Image.open(io.BytesIO(image_bytes))
        
        # Preprocessed
        processed_bytes = preprocess_for_serbian(image_bytes)
        processed = Image.open(io.BytesIO(processed_bytes))
        
        # Create comparison
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('ocr_comparison.png')
        plt.show()
        
        print("✅ Comparison saved to: ocr_comparison.png")
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")

def inspect_preprocessing_steps(image_bytes):
    """Show each preprocessing step"""
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Open image
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Apply each step and save
        steps = {}
        
        # Original
        steps['Original'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps['Grayscale'] = gray
        
        # Denoised
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        steps['Denoised'] = denoised
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        steps['Contrast Enhanced'] = enhanced
        
        # Binary
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        steps['Binary (OTSU)'] = binary
        
        # Cleaned
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        steps['Final Cleaned'] = cleaned
        
        # Plot all steps
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, step_img) in enumerate(steps.items()):
            if i < len(axes):
                if len(step_img.shape) == 2:
                    axes[i].imshow(step_img, cmap='gray')
                else:
                    axes[i].imshow(step_img)
                axes[i].set_title(name)
                axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(len(steps), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_steps.png', dpi=150)
        plt.show()
        
        print("✅ Preprocessing steps saved to: preprocessing_steps.png")
        
    except Exception as e:
        print(f"❌ Error in step inspection: {e}")

def preprocess_for_serbian(image_bytes):
    """Specialized preprocessing for Serbian documents - with auto-rotation"""
    try:
        # ===== STEP 0: AUTO-ROTATION =====
        # Open image with PIL first
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Check if image is landscape (wider than tall) - common with phone photos
        if pil_image.width > pil_image.height:
            print("🔄 Detected landscape orientation, rotating 90° clockwise")
            pil_image = pil_image.rotate(-90, expand=True)  # Negative for clockwise
        
        # Also try to detect if image is upside down
        # Convert to OpenCV for analysis
        img_initial = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray_initial = cv2.cvtColor(img_initial, cv2.COLOR_BGR2GRAY)
        
        # Check where text is concentrated (text usually at top)
        height = gray_initial.shape[0]
        top_half = gray_initial[:height//2, :]
        bottom_half = gray_initial[height//2:, :]
        
        # Text areas have more edges
        top_edges = cv2.Canny(top_half, 50, 150).sum()
        bottom_edges = cv2.Canny(bottom_half, 50, 150).sum()
        
        # If more edges at bottom, image might be upside down
        if bottom_edges > top_edges * 1.5:
            print("🔄 Detected upside down, rotating 180°")
            pil_image = pil_image.rotate(180, expand=True)
        
        # Convert back to bytes for rest of preprocessing
        rotated_bytes = io.BytesIO()
        pil_image.save(rotated_bytes, format='PNG')
        rotated_bytes.seek(0)
        image_bytes = rotated_bytes.getvalue()
        
        # ===== REST OF YOUR EXISTING PREPROCESSING =====
        # Open the (now correctly oriented) image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Increase resolution if too small
        height, width = img.shape[:2]
        if width < 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 3: LIGHT PRESERVATION - division normalization
        kernel_size = max(51, min(gray.shape) // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        normalized = cv2.divide(gray, background, scale=255)
        
        # Step 4: Gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        enhanced = clahe.apply(normalized)
        
        # Step 5: Mild denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=5, templateWindowSize=5, searchWindowSize=15)
        
        # Step 6: Gentle sharpening
        kernel = np.array([[-0.5,-0.5,-0.5],
                           [-0.5, 5,-0.5],
                           [-0.5,-0.5,-0.5]]) * 0.5
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Step 7: Dual thresholding
        _, binary_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_adaptive = cv2.adaptiveThreshold(sharpened, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 31, 5)
        binary = cv2.addWeighted(binary_otsu, 0.4, binary_adaptive, 0.6, 0)
        
        # Step 8: Light morphological cleaning
        kernel = np.ones((1,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Convert back to PIL
        result = Image.fromarray(cleaned)
        
        # Save to bytes
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return image_bytes

def detect_script(text_sample):

    return 'cyrillic'

    """Detect if text is primarily Latin or Cyrillic"""
    if not text_sample:
        return 'unknown'
    
    latin_chars = set('abcdefghijklmnopqrstuvwxyzčćžšđABCČĆŽŠĐ')
    cyrillic_chars = set('абвгдђежзијклљмнњопрстћуфхцчџшАБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ')
    
    # Count characters in each script
    latin_count = sum(1 for c in text_sample if c in latin_chars)
    cyrillic_count = sum(1 for c in text_sample if c in cyrillic_chars)
    
    if latin_count > cyrillic_count:
        return 'latin'
    elif cyrillic_count >= latin_count:
        return 'cyrillic'
    else:
        return 'unknown'

def ocr_with_script_detection(image_bytes):
    """OCR with automatic script detection and precision filtering"""
    try:
        # Preprocess image
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # ===== STRICT CYRILLIC WHITELIST =====
        # Only actual Serbian Cyrillic characters + common punctuation
        cyrillic_whitelist = "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШабвгдђежзијклљмнњопрстћуфхцчџш0123456789.,!?;:-()_ "
        
        # Build config with whitelist
        config = f'--oem 3 --psm 1 -c tessedit_char_whitelist="{cyrillic_whitelist}"'
        
        # First pass: Get data with confidence scores
        try:
            import pandas as pd
            from pytesseract import Output
            
            # Get detailed output with confidence
            ocr_data = pytesseract.image_to_data(img, lang='srp', config=config, output_type=Output.DICT)
            
            # Filter by confidence (only keep high-confidence words)
            min_confidence = 60  # Adjust this threshold (0-100)
            
            filtered_text = []
            for i, conf in enumerate(ocr_data['conf']):
                if conf != '-1' and int(conf) > min_confidence:
                    word = ocr_data['text'][i]
                    if word.strip():
                        filtered_text.append(word)
            
            text = ' '.join(filtered_text)
            
            # If filtering removed everything, fall back to regular OCR
            if len(text.strip()) < 10:
                print(f"⚠️ Low confidence filtering removed too much, falling back")
                text = pytesseract.image_to_string(img, lang='srp', config=config)
                
        except ImportError:
            # pandas not available, use regular OCR with whitelist
            print("⚠️ pandas not installed, using regular OCR")
            text = pytesseract.image_to_string(img, lang='srp', config=config)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def ocr_with_multiple_psm(image_bytes):
    """Try different page segmentation modes and return best result"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Test different PSM modes
        psm_modes = [3, 4, 6, 7, 8, 11, 13]
        results = {}
        
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm}'
            text = pytesseract.image_to_string(img, lang='srp+srp_latn', config=config)
            results[psm] = text
            logger.info(f"PSM {psm}: {len(text)} chars")
        
        # Return the result with most characters (usually best)
        best_psm = max(results, key=lambda k: len(results[k]))
        logger.info(f"Best PSM mode: {best_psm}")
        
        return results[best_psm].strip()
        
    except Exception as e:
        logger.error(f"PSM OCR error: {e}")
        return ""

def simple_ocr(image_bytes, lang='srp+srp_latn'):
    """Simple OCR with preprocessing"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        text = pytesseract.image_to_string(
            img,
            lang=lang,
            config='--oem 3 --psm 1'
        )
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Simple OCR error: {e}")
        return ""

def batch_ocr(image_paths, output_file="ocr_results.txt"):
    """Process multiple images and save results"""
    results = {}
    
    for i, path in enumerate(image_paths):
        logger.info(f"Processing {i+1}/{len(image_paths)}: {path}")
        
        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
            
            # Try script detection first
            text = ocr_with_script_detection(img_bytes)
            
            # If poor results, try multiple PSM
            if len(text) < 50:
                text = ocr_with_multiple_psm(img_bytes)
            
            results[path] = text
            
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            results[path] = f"[ERROR: {e}]"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for path, text in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"FILE: {path}\n")
            f.write(f"{'='*60}\n")
            f.write(text)
            f.write("\n\n")
    
    logger.info(f"Results saved to {output_file}")
    return results

def debug_ocr_detections(image_bytes):
    """Visualize what Tesseract detects with bounding boxes"""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
        from pytesseract import Output
        
        # Preprocess
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Convert to OpenCV for drawing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            img, 
            lang='srp', 
            config='--oem 3 --psm 1', 
            output_type=Output.DICT
        )
        
        # Create visualization
        n_boxes = len(ocr_data['text'])
        confidence_thresholds = [0, 30, 60, 90]
        
        for threshold in confidence_thresholds:
            vis_img = img_cv.copy()
            high_conf_count = 0
            
            for i in range(n_boxes):
                conf = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if conf > threshold and text:
                    high_conf_count += 1
                    (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], 
                                   ocr_data['width'][i], ocr_data['height'][i])
                    
                    # Color by confidence
                    if conf > 90:
                        color = (0, 255, 0)  # Green - high
                    elif conf > 60:
                        color = (255, 255, 0)  # Yellow - medium
                    else:
                        color = (0, 255, 255)  # Cyan - low
                    
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_img, f"{conf}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Save visualization
            output_path = f"ocr_detections_conf_{threshold}.png"
            cv2.imwrite(output_path, vis_img)
            print(f"✅ Saved: {output_path} with {high_conf_count} detections above {threshold} confidence")
        
        # Also save text with confidence
        with open("ocr_debug_output.txt", "w", encoding="utf-8") as f:
            for i in range(n_boxes):
                conf = ocr_data['conf'][i]
                text = ocr_data['text'][i]
                if text.strip():
                    f.write(f"Conf: {conf:5s} | Text: {text}\n")
        
        print("✅ Saved detailed debug to ocr_debug_output.txt")
        
    except Exception as e:
        print(f"Debug error: {e}")


def ocr_confidence_only(image_bytes, min_confidence=60):
    """OCR with ONLY confidence filtering, no character restrictions"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        from pytesseract import Output
        ocr_data = pytesseract.image_to_data(
            img, 
            lang='srp', 
            config='--oem 3 --psm 1',  # NO whitelist
            output_type=Output.DICT
        )
        
        words = []
        for i, conf in enumerate(ocr_data['conf']):
            if conf == '-1':
                continue
                
            conf_val = int(conf)
            text = ocr_data['text'][i].strip()
            
            if conf_val >= min_confidence and text:
                words.append(text)
        
        print(f"✅ Confidence-only: {len(words)} words")
        return ' '.join(words).strip()
        
    except Exception as e:
        print(f"Error: {e}")
        return ""


def ocr_with_confidence_filter(image_bytes, min_confidence=60, min_word_length=2):
    """OCR that only returns text with confidence >= threshold"""
    try:
        # Preprocess image
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Strict Serbian Cyrillic whitelist
        whitelist = "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ" + \
                    "абвгдђежзијклљмнњопрстћуфхцчџш" + \
                    "0123456789" + \
                    ".,!?;:-() " + \
                    "[]{}|\\/\"'`~@#$%^&*_+="  # Add common OCR garbage temporarily
        config = f'--oem 3 --psm 1 -c tessedit_char_whitelist="{whitelist}"'
        
        # Get detailed OCR data
        from pytesseract import Output
        ocr_data = pytesseract.image_to_data(
            img, lang='srp', config=config, output_type=Output.DICT
        )
        
        # Filter by confidence and word length
        words = []
        confs = []
        
        for i, conf in enumerate(ocr_data['conf']):
            if conf == '-1':
                continue
                
            conf_val = int(conf)
            word = ocr_data['text'][i].strip()
            
            # Apply filters
            if (word and 
                conf_val >= min_confidence and 
                len(word) >= min_word_length):
                
                # Optional: Check if word contains only valid chars
                if all(c in whitelist for c in word):
                    words.append(word)
                    confs.append(conf_val)
        
        # Log stats
        print(f"\n📊 Confidence Filter Results (min={min_confidence}%):")
        print(f"   Words kept: {len(words)}")
        if confs:
            print(f"   Avg confidence: {sum(confs)/len(confs):.1f}%")
            print(f"   Min/Max: {min(confs)}% / {max(confs)}%")
        
        # Show words with confidence for debugging
        if words:
            print("\n   Words with confidence:")
            for word, conf in zip(words[:10], confs[:10]):  # Show first 10
                print(f"     {conf:3d}%: {word}")
            if len(words) > 10:
                print(f"     ... and {len(words)-10} more")
        
        return ' '.join(words).strip()
        
    except Exception as e:
        logger.error(f"Confidence-based OCR error: {e}")
        return ""
    

def test_confidence_thresholds(image_bytes):
    """Test OCR with different confidence thresholds"""
    thresholds = [30, 40, 50, 60, 70, 80, 90]
    results = {}
    
    print("\n" + "="*60)
    print("🔬 TESTING CONFIDENCE THRESHOLDS")
    print("="*60)
    
    for threshold in thresholds:
        text = ocr_confidence_only(image_bytes, min_confidence=threshold)
        results[threshold] = {
            'text': text,
            'length': len(text),
            'words': len(text.split()) if text else 0
        }
        print(f"\n📌 Threshold {threshold}%: {results[threshold]['words']} words, {results[threshold]['length']} chars")
    
    # Save comparison
    with open("confidence_threshold_comparison.txt", "w", encoding="utf-8") as f:
        f.write("CONFIDENCE THRESHOLD COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        for threshold in thresholds:
            f.write(f"\n{'='*40}\n")
            f.write(f"THRESHOLD: {threshold}%\n")
            f.write(f"Words: {results[threshold]['words']}\n")
            f.write(f"Characters: {results[threshold]['length']}\n")
            f.write(f"{'='*40}\n")
            f.write(results[threshold]['text'])
            f.write("\n\n")
    
    print(f"\n✅ Comparison saved to confidence_threshold_comparison.txt")
    return results


def ocr_with_confidence_filter_visual(image_bytes, min_confidence=60):
    """OCR with confidence filtering AND visual output"""
    try:
        # Preprocess
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Get OCR data with boxes
        from pytesseract import Output
        ocr_data = pytesseract.image_to_data(
            img, 
            lang='srp', 
            config='--oem 3 --psm 1', 
            output_type=Output.DICT
        )
        
        # Collect all words with their boxes
        all_words = []
        kept_words = []
        filtered_words = []
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            conf = ocr_data['conf'][i]
            if conf == '-1':
                continue
                
            conf_val = int(conf)
            text = ocr_data['text'][i].strip()
            
            if not text:
                continue
            
            word_data = {
                'text': text,
                'conf': conf_val,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            }
            
            all_words.append(word_data)
            
            if conf_val >= min_confidence and len(text) >= 2:
                kept_words.append(word_data)
            else:
                filtered_words.append(word_data)
        
        # Draw ALL words (with different colors for kept vs filtered)
        img_all = Image.open(io.BytesIO(processed_bytes))
        img_cv = cv2.cvtColor(np.array(img_all), cv2.COLOR_RGB2BGR)
        
        # Draw filtered words in gray first
        for w in filtered_words:
            x, y, w_w, w_h = w['left'], w['top'], w['width'], w['height']
            cv2.rectangle(img_cv, (x, y), (x + w_w, y + w_h), (128, 128, 128), 1)
        
        # Draw kept words in green
        for w in kept_words:
            x, y, w_w, w_h = w['left'], w['top'], w['width'], w['height']
            cv2.rectangle(img_cv, (x, y), (x + w_w, y + w_h), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{w['text']}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Save
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        result.save(f"confidence_filter_{min_confidence}.png")
        
        print(f"📊 Confidence filter ({min_confidence}%):")
        print(f"   Total words detected: {len(all_words)}")
        print(f"   Kept: {len(kept_words)} (green boxes)")
        print(f"   Filtered: {len(filtered_words)} (gray boxes)")
        print(f"✅ Saved to confidence_filter_{min_confidence}.png")
        
        # Return text from kept words
        kept_text = ' '.join([w['text'] for w in kept_words])
        return kept_text
        
    except Exception as e:
        print(f"Error: {e}")
        return ""


def ocr_ensemble_visual(image_bytes):
    """Ensemble OCR with visual output showing which pass found each word"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        from pytesseract import Output
        import numpy as np
        
        # Run multiple passes
        passes = [
            {'name': 'P1_High', 'conf': 60, 'psm': 1, 'color': (0, 255, 0)},  # Green
            {'name': 'P2_Medium', 'conf': 40, 'psm': 1, 'color': (255, 255, 0)},  # Yellow
            {'name': 'P3_Low', 'conf': 30, 'psm': 1, 'color': (0, 255, 255)},  # Cyan
            {'name': 'P4_SingleLine', 'conf': 40, 'psm': 1, 'color': (255, 0, 255)},  # Magenta
        ]
        
        # Store all detections with their source pass
        all_detections = []
        
        for pass_config in passes:
            config = f'--oem 3 --psm {pass_config["psm"]}'
            
            ocr_data = pytesseract.image_to_data(
                img, lang='srp', config=config, output_type=Output.DICT
            )
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf == '-1':
                    continue
                    
                conf_val = int(conf)
                text = ocr_data['text'][i].strip()
                
                if conf_val >= pass_config['conf'] and text and len(text) >= 2:
                    all_detections.append({
                        'text': text,
                        'conf': conf_val,
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'pass': pass_config['name'],
                        'color': pass_config['color']
                    })
        
        # Remove near-duplicates (same area)
        unique_detections = []
        for det in all_detections:
            # Check if similar detection already exists
            is_duplicate = False
            for existing in unique_detections:
                # If boxes overlap significantly
                if (abs(det['left'] - existing['left']) < 20 and
                    abs(det['top'] - existing['top']) < 20):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if det['conf'] > existing['conf']:
                        unique_detections.remove(existing)
                        unique_detections.append(det)
                    break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        # Draw
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw each detection with its pass color
        for det in unique_detections:
            x, y, w, h = det['left'], det['top'], det['width'], det['height']
            color = det['color']
            
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_cv, f"{det['pass']}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Save
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        result.save("ensemble_result.png")
        
        # Count by pass
        from collections import Counter
        pass_counts = Counter([d['pass'] for d in unique_detections])
        
        print(f"📊 Ensemble Results:")
        for pass_name, count in pass_counts.items():
            print(f"   {pass_name}: {count} words")
        print(f"   Total unique: {len(unique_detections)}")
        print(f"✅ Saved to ensemble_result.png")
        
        # Sort by position and return text
        unique_detections.sort(key=lambda x: (x['top'], x['left']))
        return ' '.join([d['text'] for d in unique_detections])
        
    except Exception as e:
        print(f"Error: {e}")
        return ""
    

def ocr_two_pass_visual(image_bytes):
    """Two-pass OCR with visual output"""
    try:
        processed_bytes = preprocess_for_serbian(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        from pytesseract import Output
        
        # Get all OCR data
        ocr_data = pytesseract.image_to_data(
            img, lang='srp', config='--oem 3 --psm 1', output_type=Output.DICT
        )
        
        # Collect all boxes
        all_boxes = []
        for i, conf in enumerate(ocr_data['conf']):
            if conf == '-1':
                continue
                
            text = ocr_data['text'][i].strip()
            if not text:
                continue
                
            all_boxes.append({
                'text': text,
                'conf': int(conf),
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            })
        
        # Pass 1: High confidence words
        high_conf = [b for b in all_boxes if b['conf'] >= 60 and len(b['text']) >= 2]
        
        # Pass 2: Medium confidence near high confidence
        medium_conf = []
        for box in all_boxes:
            if 40 <= box['conf'] < 60 and len(box['text']) >= 2:
                # Check if near any high confidence word
                for hc in high_conf:
                    # Vertical proximity
                    if abs(box['top'] - hc['top']) < 50:
                        medium_conf.append(box)
                        break
        
        # Draw
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw all other words in gray
        kept_indices = set([id(b) for b in high_conf + medium_conf])
        for box in all_boxes:
            if id(box) not in kept_indices:
                x, y, w, h = box['left'], box['top'], box['width'], box['height']
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (128, 128, 128), 1)
        
        # Draw medium confidence in yellow
        for box in medium_conf:
            x, y, w, h = box['left'], box['top'], box['width'], box['height']
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img_cv, f"{box['text']}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw high confidence in green
        for box in high_conf:
            x, y, w, h = box['left'], box['top'], box['width'], box['height']
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{box['text']}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Save
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        result.save("two_pass_result.png")
        
        print(f"📊 Two-Pass Results:")
        print(f"   High confidence (green): {len(high_conf)}")
        print(f"   Medium confidence near text (yellow): {len(medium_conf)}")
        print(f"   Filtered (gray): {len(all_boxes) - len(high_conf) - len(medium_conf)}")
        print(f"✅ Saved to two_pass_result.png")
        
        # Return combined text
        all_kept = high_conf + medium_conf
        all_kept.sort(key=lambda x: (x['top'], x['left']))  # Sort by position
        return ' '.join([b['text'] for b in all_kept])
        
    except Exception as e:
        print(f"Error: {e}")
        return ""


def draw_word_boxes(image_bytes, words_with_boxes, output_path="ocr_boxes.png", title="OCR Results"):
    """Draw bounding boxes around detected words"""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Open original image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to OpenCV for drawing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw boxes for each word
        for item in words_with_boxes:
            word = item['text']
            conf = item.get('conf', 0)
            x, y, w, h = item['left'], item['top'], item['width'], item['height']
            
            # Color based on confidence
            if conf >= 60:
                color = (0, 255, 0)  # Green - high confidence
            elif conf >= 40:
                color = (255, 255, 0)  # Yellow - medium
            else:
                color = (0, 255, 255)  # Cyan - low
            
            # Draw rectangle
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
            
            # Add word text above box
            cv2.putText(img_cv, f"{word}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add confidence
            cv2.putText(img_cv, f"{conf}%", (x, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Convert back to PIL and save
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        result.save(output_path)
        print(f"✅ Boxed image saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return None

import time

if __name__ == "__main__":
    # Test if Tesseract can find the Serbian language
    print("🔍 Checking available Tesseract languages...")
    try:
        languages = pytesseract.get_languages()
        print(f"Available languages: {languages}")
        if 'srp' in languages:
            print("✅ Serbian language (srp) found!")
        else:
            print("❌ Serbian language (srp) NOT found. Check TESSDATA_PREFIX path.")
            print(f"   Looking in: {os.environ.get('TESSDATA_PREFIX')}")
    except Exception as e:
        print(f"❌ Error checking languages: {e}")
    
    # ... rest of your code
    
    print("="*60)
    print("🧪 SERBIAN OCR TESTING SUITE WITH VISUALIZATION")
    print("="*60)
    
    # Test image path
    image_path = r"C:\Users\cerim\ai-test-solver\data\srpski_2.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        # Try data folder
        if os.path.exists("data/srpski_2.jpg"):
            image_path = "data/srpski_2.jpg"
            print(f"✅ Found at: {image_path}")
        else:
            # Look for any JPG in data folder
            if os.path.exists("data"):
                jpg_files = [f for f in os.listdir("data") if f.endswith('.jpg')]
                if jpg_files:
                    image_path = os.path.join("data", jpg_files[0])
                    print(f"✅ Using: {image_path}")
                else:
                    print("❌ No JPG files found in data folder")
                    exit(1)
            else:
                print("❌ data folder not found")
                exit(1)
    
    # Read image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    print(f"\n📸 Testing on: {image_path}")
    print(f"File size: {len(img_bytes)} bytes")

     # ===== ADD THIS LINE HERE =====
    # Run diagnostic first
    print("\n🔍 Running OCR diagnostic...")
    debug_ocr_detections(img_bytes)
    
    # Ask user what to do
    print("\n" + "="*60)
    print("OPTIONS:")
    print("1. Run OCR tests only")
    print("2. Save preprocessed image")
    print("3. Compare original vs preprocessed")
    print("4. Inspect all preprocessing steps")
    print("5. Run everything")
    # Add option for confidence-based OCR
    print("7. Test confidence thresholds")
    print("8. Run confidence-based OCR (60%)")
    print("🎯 RECALL IMPROVEMENT OPTIONS")
    print("="*60)
    print("9. Test recall strategies")
    print("10. Run two-pass OCR")
    print("11. Run ensemble OCR")
    print("="*60)
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice in ['2', '3', '4', '5']:
        # Save preprocessed image
        print("\n📸 SAVING PREPROCESSED IMAGE...")
        saved_path = save_preprocessed_image(img_bytes, "preprocessed_output.png")
        if saved_path:
            print(f"✅ Open '{saved_path}' to see what Tesseract sees!")
    
    """
    if choice in ['3', '5']:
        # Show comparison
        print("\n🖼️ CREATING COMPARISON...")
        try:
            compare_original_vs_preprocessed(img_bytes)
        except ImportError:
            print("⚠️ matplotlib not installed. Install with: pip install matplotlib")
            # Fallback to simple save
            save_preprocessed_image(img_bytes, "preprocessed.png")
    """
    
    if choice in ['4', '5']:
        # Show all steps
        print("\n🔍 INSPECTING PREPROCESSING STEPS...")
        try:
            inspect_preprocessing_steps(img_bytes)
        except ImportError:
            print("⚠️ matplotlib not installed. Install with: pip install matplotlib")
    
    if choice in ['1', '5']:
        # Run OCR tests
        print("\n" + "="*60)
        print("📊 RUNNING OCR TESTS")
        print("="*60)
        
        # Test 1: Basic OCR
        print("\n📄 BASIC OCR:")
        start = time.time()
        img = Image.open(io.BytesIO(img_bytes))
        basic_text = pytesseract.image_to_string(img, lang='srp+eng', config='--oem 3 --psm 1')
        print(f"Time: {time.time()-start:.2f}s")
        print(f"Chars: {len(basic_text)}")
        print(f"Preview: {basic_text[:200]}")
        
        # Test 2: With preprocessing
        print("\n🔧 WITH PREPROCESSING:")
        start = time.time()
        proc_text = simple_ocr(img_bytes)
        print(f"Time: {time.time()-start:.2f}s")
        print(f"Chars: {len(proc_text)}")
        print(f"Preview: {proc_text[:200]}")
        
        # Test 3: With script detection
        print("\n🎯 WITH SCRIPT DETECTION:")
        start = time.time()
        detect_text = ocr_with_script_detection(img_bytes)
        print(f"Time: {time.time()-start:.2f}s")
        print(f"Chars: {len(detect_text)}")
        print(f"Preview: {detect_text[:200]}")
        
        # Summary
        print("\n" + "="*50)
        print("📊 SUMMARY")
        print("="*50)
        print(f"Basic OCR:        {len(basic_text):5d} chars")
        print(f"Preprocessing:    {len(proc_text):5d} chars")
        print(f"Script Detection: {len(detect_text):5d} chars")
        
        # ===== NEW CODE: Save best result =====
        print("\n" + "="*50)
        print("💾 SAVING BEST RESULT")
        print("="*50)
        
        # Compare results
        results_dict = {
            "basic": basic_text,
            "preprocessing": proc_text,
            "script_detection": detect_text
        }
        
        # Find method with most text
        best_method = max(results_dict, key=lambda k: len(results_dict[k]))
        best_text = results_dict[best_method]
        
        print(f"Best method: {best_method} ({len(best_text)} chars)")
        
        # Save to file
        output_path = "best_ocr_result.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"BEST OCR RESULT\n")
            f.write(f"Method: {best_method}\n")
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Characters: {len(best_text)}\n")
            f.write("="*60 + "\n\n")
            f.write(best_text)
        
        print(f"✅ Saved to: {output_path}")
        
        # Also save individual results for comparison
        with open("ocr_results_all.txt", "w", encoding="utf-8") as f:
            for method, text in results_dict.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"METHOD: {method}\n")
                f.write(f"Length: {len(text)} chars\n")
                f.write(f"{'='*60}\n")
                f.write(text)
                f.write("\n\n")
        
        print(f"✅ All results saved to: ocr_results_all.txt")
    
    if choice == '7':
        test_confidence_thresholds(img_bytes)
    
    if choice == '8':
        text = ocr_confidence_only(img_bytes, min_confidence=60)
        print(f"\n📝 Result ({len(text)} chars):")
        print(text)
        
        # Save to file
        with open("confidence_60_result.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("✅ Saved to confidence_60_result.txt")
    
    if choice == '9':
        text = ocr_with_confidence_filter_visual(img_bytes, min_confidence=60)
        with open("visual_confidence_60.txt", "w", encoding="utf-8") as f:
            f.write(text)

    if choice == '10':
        text = ocr_two_pass_visual(img_bytes)
        with open("visual_two_pass.txt", "w", encoding="utf-8") as f:
            f.write(text)

    if choice == '11':
        text = ocr_ensemble_visual(img_bytes)
        with open("ensemble_result.txt", "w", encoding="utf-8") as f:
            f.write(text)
    
    print("\n✅ Done! Check the generated images to see what's happening.")