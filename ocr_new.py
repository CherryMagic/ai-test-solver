"""
OCR for Serbian with:
- Auto-rotation (fixes upside-down or rotated images)
- Preprocessing (light-preserving)
- Character-level boxes (blue)
- Word-level boxes (green/yellow/orange by confidence)
- Output to text file and visualized image
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import pytesseract
import logging
import os
import re
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = os.path.join(os.getcwd(), 'tessdata')

# Optional: Custom tessdata folder
TESSDATA_PATH = os.path.join(os.getcwd(), 'tessdata')
if os.path.exists(TESSDATA_PATH):
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
    print(f"📂 Using tessdata from: {TESSDATA_PATH}")

# Fix temp directory issue
custom_temp = os.path.join(os.getcwd(), "temp")
os.makedirs(custom_temp, exist_ok=True)
tempfile.tempdir = custom_temp
print(f"📂 Using temp directory: {custom_temp}")

def auto_rotate_image(image):
    """Detect and correct image rotation using multiple methods"""
    try:
        # Convert PIL to OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # METHOD 1: Try Tesseract's orientation detection first
        try:
            # Save temporarily for OSD
            temp_img = os.path.join(custom_temp, "temp_osd.png")
            image.save(temp_img)
            osd = pytesseract.image_to_osd(temp_img, config='--psm 0')
            
            # Parse orientation
            import re
            rotate_match = re.search(r'Rotate: (\d+)', osd)
            if rotate_match:
                rotate = int(rotate_match.group(1))
                if rotate != 0:
                    print(f"🔄 Tesseract OSD: Rotate by {rotate}°")
                    rotated = image.rotate(rotate, expand=True)
                    return rotated
        except Exception as e:
            print(f"   OSD method failed: {e}")
        
        # METHOD 2: Check if image is landscape (wider than tall)
        if image.width > image.height * 1.2:  # Significantly wider
            print("🔄 Image is landscape mode, rotating to portrait")
            return image.rotate(90, expand=True)
        
        # METHOD 3: Try edge-based detection for small rotations
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            angles = []
            for line in lines:
                theta = line[0][1]
                # Convert to degrees, adjust to horizontal
                angle = theta * 180 / np.pi - 90
                # Filter near-horizontal lines (text lines)
                if abs(angle) < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 2:
                    print(f"🔄 Edge detection: Rotating by {median_angle:.1f}°")
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h), 
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
                    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        
        # If no rotation needed
        print("✅ No significant rotation detected")
        return image
        
    except Exception as e:
        print(f"Rotation error: {e}")
        return image

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


def preprocess_image(image_bytes):
    """Light-preserving preprocessing with auto-rotation"""
    try:
        # Open image
        pil_image = Image.open(io.BytesIO(image_bytes))
        print(f"📸 Original image: {pil_image.size}")
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            print(f"   Converted from {pil_image.mode} to RGB")
        
        # STEP 0: AUTO-ROTATION
        print("\n🔄 Checking orientation...")
        pil_image = auto_rotate_image(pil_image)
        
        # Check if image is landscape (wider than tall)
        if pil_image.width > pil_image.height:
            print("🔄 Image is landscape, rotating to portrait")
            pil_image = pil_image.rotate(-90, expand=True)
        
        print(f"   After rotation: {pil_image.size}")
        
        # Save intermediate for debugging
        pil_image.save("after_rotation.png")
        print("✅ Saved after_rotation.png")
        
        # Convert to OpenCV
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Increase resolution if too small
        height, width = img.shape[:2]
        if width < 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"📏 Resized from {width}x{height} to {new_width}x{new_height}")
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Shadow removal
        kernel_size = max(51, min(gray.shape) // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        normalized = cv2.divide(gray, background, scale=255)
        
        # Step 4: Gentle contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        enhanced = clahe.apply(normalized)
        
        # Step 5: Mild denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=5, templateWindowSize=5, searchWindowSize=15)
        
        # Step 6: Convert back to PIL
        result = Image.fromarray(denoised)
        
        # Save to bytes
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        return output.getvalue()
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return image_bytes

def ocr_with_boxes(image_bytes, lang='srp', psm=1):
    """
    OCR with visual boxes showing:
    - Character-level boxes (blue)
    - Word-level boxes (green/yellow/orange by confidence)
    """
    try:
        # Step 1: Preprocess (with auto-rotation)
        print("\n🔧 Preprocessing image...")
        processed_bytes = preprocess_image(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Save preprocessed image
        img.save("preprocessed.png")
        print("✅ Preprocessed image saved to: preprocessed.png")
        
        # Step 2: Get character-level boxes
        print(f"\n🔍 Running OCR with PSM {psm}...")
        
        # Get character boxes (might fail, but we'll try)
        char_data = None
        try:
            char_data = pytesseract.image_to_boxes(img, lang=lang, config=f'--psm {psm}')
            print("✅ Character-level data retrieved")
        except Exception as e:
            print(f"⚠️ Character boxes failed: {e}")
        
        # Get word data with confidence
        from pytesseract import Output
        word_data = pytesseract.image_to_data(
            img, 
            lang=lang, 
            config=f'--psm {psm}', 
            output_type=Output.DICT
        )
        
        # Get full text
        full_text = pytesseract.image_to_string(img, lang=lang, config=f'--psm {psm}')
        
        # Step 3: Create visualization
        # Convert PIL to OpenCV for drawing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h_img, w_img = img_cv.shape[:2]
        
        # Draw character boxes if available (blue)
        if char_data:
            print("\n📦 Drawing character boxes...")
            char_count = 0
            for line in char_data.splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    char = parts[0]
                    x1 = int(parts[1])
                    y1 = h_img - int(parts[2])  # Tesseract coordinates are from bottom
                    x2 = int(parts[3])
                    y2 = h_img - int(parts[4])
                    
                    # Draw blue box for each character
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    char_count += 1
                    
                    # Put character (commented to reduce clutter)
                    # if len(char) == 1 and char.isalpha():
                    #     cv2.putText(img_cv, char, (x1, y1-2), 
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            print(f"   Total character boxes: {char_count}")
        
        # Draw word boxes with confidence
        print("\n📦 Drawing word boxes...")
        word_count = 0
        high_conf_words = []
        medium_conf_words = []
        low_conf_words = []
        
        # Store words by confidence for statistics
        all_words = []
        
        for i, text in enumerate(word_data['text']):
            conf = word_data['conf'][i]
            if conf != '-1' and text.strip():
                conf_val = int(conf)
                word = text.strip()
                x = word_data['left'][i]
                y = word_data['top'][i]
                w = word_data['width'][i]
                h = word_data['height'][i]
                
                word_count += 1
                all_words.append((word, conf_val))
                
                # Color based on confidence
                if conf_val >= 60:
                    color = (0, 255, 0)  # Green - high confidence
                    high_conf_words.append(word)
                elif conf_val >= 30:
                    color = (0, 255, 255)  # Yellow - medium confidence
                    medium_conf_words.append(word)
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                    low_conf_words.append(word)
                
                # Draw word box
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
                
                # Add confidence text
                cv2.putText(img_cv, f"{conf_val}%", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw approximate character divisions
                if len(word) > 1:
                    char_width = w // max(1, len(word))
                    for j in range(1, len(word)):
                        char_x = x + j * char_width
                        cv2.line(img_cv, (char_x, y), (char_x, y + h), (255, 0, 0), 1)
        
        print(f"   Total word boxes: {word_count}")
        print(f"   High confidence (≥60%): {len(high_conf_words)}")
        print(f"   Medium confidence (30-60%): {len(medium_conf_words)}")
        print(f"   Low confidence (<30%): {len(low_conf_words)}")
        
        # Calculate average confidence
        conf_values = [c for _, c in all_words if c > 0]
        if conf_values:
            avg_conf = sum(conf_values) / len(conf_values)
            print(f"   Average confidence: {avg_conf:.1f}%")
        
        # Add legend
        cv2.putText(img_cv, "BLUE: Character boxes / divisions", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img_cv, "GREEN: Word (≥60% conf)", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_cv, "YELLOW: Word (30-60% conf)", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img_cv, "ORANGE: Word (<30% conf)", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Save visualization
        cv2.imwrite("ocr_boxes_psm1.png", img_cv)
        print("\n✅ Saved visualization to: ocr_boxes_psm1.png")
        
        return full_text, word_data
        
    except Exception as e:
        print(f"Error in OCR: {e}")
        import traceback
        traceback.print_exc()
        return "", None

def test_ocr(image_path):
    """Main test function"""
    
    print("="*70)
    print("🧪 SERBIAN OCR WITH AUTO-ROTATION AND BOXES")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    # Read image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    print(f"\n📸 Image: {image_path}")
    print(f"   Size: {len(img_bytes)} bytes")
    
    # Run OCR with boxes
    text, word_data = ocr_with_boxes(
        img_bytes, 
        lang='srp', 
        psm=1
    )
    
    # Show results
    print("\n" + "="*70)
    print("📝 OCR RESULT")
    print("="*70)
    print(text)
    
    print("\n" + "="*70)
    print("📊 STATISTICS")
    print("="*70)
    print(f"Total characters: {len(text)}")
    print(f"Total words: {len(text.split())}")
    print(f"Total lines: {len(text.splitlines())}")
    
    # Save text to file
    output_txt = "ocr_result.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("OCR RESULT\n")
        f.write(f"Image: {image_path}\n")
        f.write("="*60 + "\n\n")
        f.write(text)
    
    print(f"\n💾 Text saved to: {output_txt}")
    
    # Also save detailed debug info
    debug_txt = "ocr_debug.txt"
    with open(debug_txt, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("OCR DEBUG INFO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Characters: {len(text)}\n")
        f.write(f"Words: {len(text.split())}\n")
        
        if word_data:
            f.write("\n--- Word Confidence Breakdown ---\n")
            high, medium, low = 0, 0, 0
            for i, t in enumerate(word_data['text']):
                if t.strip():
                    conf = word_data['conf'][i]
                    if conf != '-1':
                        conf_val = int(conf)
                        f.write(f"  {conf_val:3d}% : {t}\n")
                        if conf_val >= 60:
                            high += 1
                        elif conf_val >= 30:
                            medium += 1
                        else:
                            low += 1
            f.write(f"\nSummary: High={high}, Medium={medium}, Low={low}\n")
    
    print(f"💾 Debug info saved to: {debug_txt}")
    
    print("\n✅ Generated files:")
    print("   - after_rotation.png (image after rotation)")
    print("   - preprocessed.png (image after preprocessing)")
    print("   - ocr_boxes_psm1.png (visualization with boxes)")
    print("   - ocr_result.txt (extracted text)")
    print("   - ocr_debug.txt (detailed debug info)")
    
    return text

if __name__ == "__main__":
    import os
    
    # Default test image - CHANGE THIS TO YOUR IMAGE
    image_path = r"C:\Users\cerim\ai-test-solver\data\srpski_1.jpg"
    
    # If default not found, look for any image in data folder
    if not os.path.exists(image_path):
        if os.path.exists("data"):
            images = [f for f in os.listdir("data") if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                image_path = os.path.join("data", images[0])
                print(f"✅ Found image: {image_path}")
            else:
                print("❌ No images found in data folder")
                
                # Ask user for path
                image_path = input("Enter path to image file: ").strip()
                if not os.path.exists(image_path):
                    print("❌ File not found")
                    exit(1)
        else:
            print("❌ data folder not found")
            # Ask user for path
            image_path = input("Enter path to image file: ").strip()
            if not os.path.exists(image_path):
                print("❌ File not found")
                exit(1)
    
    # Run test
    test_ocr(image_path)