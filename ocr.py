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
        config = f'--oem 3 --psm 6 -c tessedit_char_whitelist="{cyrillic_whitelist}"'
        
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
            config='--oem 3 --psm 6'
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
    
    # Ask user what to do
    print("\n" + "="*60)
    print("OPTIONS:")
    print("1. Run OCR tests only")
    print("2. Save preprocessed image")
    print("3. Compare original vs preprocessed")
    print("4. Inspect all preprocessing steps")
    print("5. Run everything")
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
        basic_text = pytesseract.image_to_string(img, lang='srp+eng', config='--oem 3 --psm 6')
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
    
    print("\n✅ Done! Check the generated images to see what's happening.")