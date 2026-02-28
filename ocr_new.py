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


UNDERLINE_CHAR = '▁' 


class EmptyLineDetector:
    """Detects empty lines and blank answer spaces in images"""
    
    def __init__(self):
        self.line_height_threshold = 15  # Minimum height for a line of text
        self.empty_space_ratio = 0.9     # If 90% of a row is white, consider it empty
        self.min_blank_width = 30         # Minimum width for a blank answer space
        self.min_blank_height = 10        # Minimum height for a blank answer space
        
    def detect_empty_lines(self, img_array, visualize=True):
        """
        Detect empty lines in an image using horizontal projection
        Returns: list of (y_start, y_end) for empty lines
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Binarize (0 = black, 255 = white)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Calculate horizontal projection (sum of black pixels per row)
        # Black pixels (text) have value 0, so we invert
        inverted = 255 - binary
        horizontal_projection = np.sum(inverted, axis=1) / inverted.shape[1]
        
        # Normalize
        max_proj = np.max(horizontal_projection)
        if max_proj > 0:
            horizontal_projection = horizontal_projection / max_proj
        
        # Find empty regions (where projection is below threshold)
        empty_threshold = 0.05  # Less than 5% of max text density
        is_empty = horizontal_projection < empty_threshold
        
        # Group consecutive empty rows into lines
        empty_lines = []
        in_empty = False
        start_row = 0
        
        for i, empty in enumerate(is_empty):
            if empty and not in_empty:
                # Start of empty region
                in_empty = True
                start_row = i
            elif not empty and in_empty:
                # End of empty region
                in_empty = False
                if i - start_row >= self.line_height_threshold:
                    empty_lines.append((start_row, i))
        
        # Check end of image
        if in_empty and len(is_empty) - start_row >= self.line_height_threshold:
            empty_lines.append((start_row, len(is_empty)))
        
        return empty_lines, horizontal_projection
    
    def detect_answer_blanks(self, img_array):
        """
        Detect blank answer spaces (_____ style) using contour detection
        Returns: list of (x, y, w, h) for blank spaces
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Threshold to find dark areas (underscores are dark)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blanks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for underscore-like shapes: wide and short
            if (w > self.min_blank_width and 
                h < self.min_blank_height * 3 and
                h > self.min_blank_height / 2 and
                w > h * 3):  # At least 3x wider than tall
                blanks.append((x, y, w, h))
        
        return blanks
    
    def detect_multiple_choice_circles(self, img_array):
        """
        Detect circles for multiple choice questions
        Returns: list of (x, y, r) for circles
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=30
        )
        
        circle_list = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circle_list.append((x, y, r))
        
        return circle_list
    
    def detect_underlines_as_characters(self, img_array):
        """
        Detect underline lines
        Returns: list of (x1, y1, x2, y2) for underlines
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=50, 
            maxLineGap=5
        )
        
        underline_chars = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 20:  # Nearly horizontal
                    # Create a bounding box around the line
                    x = min(x1, x2)
                    y = min(y1, y2) - 5  # Extend upward slightly
                    w = abs(x2 - x1)
                    h = 10  # Fixed height for underline character
                    
                    # Split long underlines into multiple characters
                    # (each underline character will be about 30-50 pixels wide)
                    char_width = 40
                    if w > char_width:
                        num_chars = w // char_width
                        for i in range(num_chars):
                            char_x = x + i * char_width
                            char_w = min(char_width, w - i * char_width)
                            if char_w > 20:  # Minimum width to count
                                underline_chars.append((char_x, y, char_w, h, '▁'))
                    else:
                        underline_chars.append((x, y, w, h, '▁'))
        
        return underline_chars



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
        # pil_image = auto_rotate_image(pil_image)
        
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
    OCR with natural underscore detection (no whitelist)
    Returns: text with proper spacing and underscores
    """
    try:
        # Step 1: Preprocess (with auto-rotation)
        print("\n🔧 Preprocessing image...")
        processed_bytes = preprocess_image(image_bytes)
        img = Image.open(io.BytesIO(processed_bytes))
        
        # Save preprocessed image
        img.save("preprocessed.png")
        print("✅ Preprocessed image saved to: preprocessed.png")
        
        # ===== OCR WITHOUT WHITELIST =====
        from pytesseract import Output
        
        # Simple config - no whitelist, just PSM
        custom_config = f'--psm {psm}'
        
        print(f"\n🔍 Running OCR with PSM {psm} (no whitelist)...")
        
        # Get word data with confidence and positions
        word_data = pytesseract.image_to_data(
            img, 
            lang=lang, 
            config=custom_config, 
            output_type=Output.DICT
        )
        
        # Get full text for comparison
        full_text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
        print(f"\n📝 Raw Tesseract output:\n{full_text}")
        
        # Build properly spaced text with positions
        words_with_pos = []
        
        for i, text in enumerate(word_data['text']):
            conf = word_data['conf'][i]
            if conf != '-1' and text.strip():
                words_with_pos.append({
                    'text': text.strip(),
                    'conf': int(conf),
                    'x': word_data['left'][i],
                    'y': word_data['top'][i]
                })
        
        # Sort by vertical (y) then horizontal (x)
        words_with_pos.sort(key=lambda w: (w['y'], w['x']))
        
        # Group into lines
        lines = []
        current_line = []
        last_y = None
        y_threshold = 15  # Adjust based on your image
        
        for w in words_with_pos:
            if last_y is None or abs(w['y'] - last_y) <= y_threshold:
                current_line.append(w)
            else:
                # Sort current line by x and join with spaces
                current_line.sort(key=lambda w: w['x'])
                line_text = ' '.join([w['text'] for w in current_line])
                lines.append(line_text)
                current_line = [w]
            last_y = w['y']
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda w: w['x'])
            line_text = ' '.join([w['text'] for w in current_line])
            lines.append(line_text)
        
        # Final text with proper spacing
        final_text = '\n'.join(lines)
        
        # Create visualization with boxes
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw word boxes with confidence
        print("\n📦 Drawing word boxes...")
        word_count = 0
        underscore_words = []
        
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
                
                # Check if word contains underscore
                if '_' in word:
                    underscore_words.append(word)
                    color = (255, 0, 255)  # Purple for words with underscore
                else:
                    # Color based on confidence
                    if conf_val >= 60:
                        color = (0, 255, 0)  # Green
                    elif conf_val >= 30:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                
                # Draw word box
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
                
                # Add confidence text
                cv2.putText(img_cv, f"{conf_val}%", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # If underscore word, mark it
                if '_' in word:
                    cv2.putText(img_cv, "HAS_UNDERSCORE", (x, y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
        
        print(f"   Total word boxes: {word_count}")
        print(f"   Words with underscores: {len(underscore_words)}")
        if underscore_words:
            print(f"   Examples: {underscore_words[:5]}")
        
        # Add legend
        y_pos = 30
        cv2.putText(img_cv, "GREEN: Word (≥60% conf)", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_cv, "YELLOW: Word (30-60% conf)", (10, y_pos+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img_cv, "ORANGE: Word (<30% conf)", (10, y_pos+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(img_cv, "PURPLE: Word contains underscore '_'", (10, y_pos+60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Save visualization
        cv2.imwrite("ocr_boxes_underscore.png", img_cv)
        print("\n✅ Saved visualization to: ocr_boxes_underscore.png")
        
        # Show sample with underscores
        print(f"\n📝 Sample text with underscores:")
        for line in lines[:5]:  # Show first 5 lines
            if '_' in line:
                print(f"   🔍 {line}")
            else:
                print(f"      {line}")
        
        # Count underscores in final text
        total_underscores = final_text.count('_')
        print(f"\n📊 Total underscores in text: {total_underscores}")
        
        return final_text, word_data
        
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

        f.write("\n\n" + "="*60 + "\n")
        f.write("DOCUMENT STRUCTURE\n")
        f.write("="*60 + "\n")
    
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
    image_path = r"C:\Users\cerim\ai-test-solver\data\pid_1.jpg"
    
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