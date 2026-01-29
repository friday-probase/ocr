"""
PAYSLIP NET PAY EXTRACTION SERVICE
Uses LLM for extracting net pay from payslips
Supports: Ollama (self-hosted) or Groq (free cloud)
File types: PNG, JPG, JPEG, PDF
"""

import json
import time
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from typing import Dict, Any, Optional, List
import os
import re
from pdf2image import convert_from_path
import shutil


# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Tesseract config optimized for payslips (tabular documents)
TESSERACT_CONFIG = '--oem 3 --psm 6'  # Block text detection

# LLM Configuration - supports multiple providers
# Provider options: 'groq' (default, free cloud), 'ollama' (self-hosted)
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq')

# Ollama config (self-hosted)
OLLAMA_MODEL = os.getenv('LLM_MODEL', 'qwen2.5:1.5b')
OLLAMA_ENDPOINT = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')

# Groq config (free cloud - get key at console.groq.com)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')  # Fast and free
GROQ_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'


LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '120'))


def call_llm(prompt: str) -> Optional[str]:
    """
    Call LLM based on configured provider (Ollama or Groq)
    Returns the response text or None on error
    """
    provider = LLM_PROVIDER.lower()
    
    if provider == 'groq':
        return call_groq(prompt)
    else:
        return call_ollama(prompt)


def call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama API"""
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 500,
                    'num_ctx': 8192
                }
            },
            timeout=LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return None


def call_groq(prompt: str) -> Optional[str]:
    """Call Groq API (free, fast inference)"""
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Get free key at console.groq.com")
        return None
    
    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': GROQ_MODEL,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.1,
                'max_tokens': 500
            },
            timeout=LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Groq error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Groq call failed: {e}")
        return None


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy for payslips
    - Convert to grayscale
    - Increase contrast  
    - Sharpen edges
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast (payslips often have low contrast)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.7)
    
    # Sharpen for better text detection
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Apply slight smoothing to reduce noise
    image = image.filter(ImageFilter.SMOOTH)
    
    return image


def detect_and_rotate_image(image: Image.Image) -> Image.Image:
    """
    Detect image orientation using Tesseract OSD and rotate if needed
    """
    width, height = image.size
    
    try:
        # Preprocess first for better OSD results
        processed = preprocess_image_for_ocr(image)
        
        # Get orientation info from Tesseract
        osd = pytesseract.image_to_osd(processed)
        
        # Parse the rotation angle
        angle = 0
        for line in osd.split('\n'):
            if 'Rotate:' in line:
                angle = int(line.split(':')[1].strip())
                break
        
        if angle != 0:
            print(f"  🔄 Rotating image by {angle}°...")
            image = image.rotate(-angle, expand=True)
            
        return image
            
    except Exception as e:
        print(f"  ⚠ OSD detection failed: {e}")
        
        # Fallback: Manual landscape detection
        if width > height:
            # Try clockwise rotation for landscape
            return image.rotate(-90, expand=True)
        
        return image


def load_images_from_file(file_path: str) -> List[Image.Image]:
    """
    Load all images from file, converting multi-page PDF if necessary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return []
    
    # Handle PDF files
    if file_path.suffix.lower() == '.pdf':
        try:
            print(f"Converting PDF to images: {file_path}")
            
            # Find poppler path
            poppler_path = None
            if os.name == 'nt':  # Windows
                poppler_path = r'C:\poppler\poppler-24.08.0\Library\bin'
            else:
                pdftoppm_location = shutil.which('pdftoppm')
                if pdftoppm_location:
                    poppler_path = os.path.dirname(pdftoppm_location)
            
            # Convert PDF to images at 300 DPI (optimized for speed vs quality)
            images = convert_from_path(str(file_path), dpi=300, poppler_path=poppler_path)
            
            if images:
                print(f"PDF converted successfully ({len(images)} page(s))")
                # Auto-rotate each page
                rotated_images = []
                for i, img in enumerate(images, 1):
                    rotated_img = detect_and_rotate_image(img)
                    rotated_images.append(rotated_img)
                return rotated_images
            else:
                print("ERROR: PDF conversion returned no images")
                return []
                
        except Exception as e:
            print(f"ERROR: Failed to convert PDF: {e}")
            return []
    
    # Handle regular image files
    try:
        image = Image.open(file_path)
        print(f"Image loaded: {file_path.suffix.upper()}, size: {image.size}")
        image = detect_and_rotate_image(image)
        return [image]
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        return []


def load_images_from_bytes(file_bytes: bytes, filename: str = "document") -> List[Image.Image]:
    """
    Load all images from bytes (for API usage)
    """
    import io
    import tempfile
    
    # Detect if PDF by checking magic bytes
    is_pdf = file_bytes[:4] == b'%PDF'
    
    if is_pdf:
        # Save to temp file for pdf2image
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            images = load_images_from_file(tmp_path)
        finally:
            os.unlink(tmp_path)
        return images
    else:
        # Load as image
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image = detect_and_rotate_image(image)
            return [image]
        except Exception as e:
            print(f"ERROR: Failed to load image from bytes: {e}")
            return []


def perform_ocr(image: Image.Image) -> str:
    """
    Perform OCR using Tesseract (lightweight, no GPU required)
    """
    try:
        processed_image = preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(processed_image, config=TESSERACT_CONFIG)
        return text
    except Exception as e:
        print(f"  ⚠ OCR failed: {e}")
        return ""


def extract_payslip_llm(file_path: str = None, file_bytes: bytes = None) -> Dict[str, Any]:
    """
    Extract net pay from payslip using LLM
    
    Args:
        file_path: Path to payslip image/PDF (use either this OR file_bytes)
        file_bytes: Raw bytes of the file
    
    Returns:
        Dictionary with:
        - success (bool): Whether extraction succeeded
        - data (dict): Extracted payslip data including net_pay
        - confidence (float): Extraction confidence
        - processing_time_seconds (float): Processing time
        - errors (list): Any errors encountered
    """
    start_time = time.time()
    result = {
        'success': False,
        'data': {},
        'confidence': 0.0,
        'processing_time_seconds': 0.0,
        'errors': []
    }
    
    try:
        # Step 1: Load images
        if file_path:
            images = load_images_from_file(file_path)
        elif file_bytes:
            images = load_images_from_bytes(file_bytes)
        else:
            result['errors'].append("No file_path or file_bytes provided")
            return result
        
        if not images:
            result['errors'].append("Failed to load images from file")
            return result
        
        # Step 2: OCR all pages
        all_ocr_text = []
        for i, image in enumerate(images, 1):
            print(f"Processing page {i}/{len(images)}...")
            page_text = perform_ocr(image)
            if page_text.strip():
                all_ocr_text.append(f"--- PAGE {i} ---\n{page_text}")
        
        if not all_ocr_text:
            result['errors'].append("OCR returned no text from any page")
            return result
        
        ocr_text = "\n\n".join(all_ocr_text)
        
        # Log OCR results summary
        print(f"OCR completed: {len(ocr_text)} characters from {len(images)} page(s)")
        
        # Step 3: LLM Extraction
        prompt = f"""You are a payslip data extraction AI. Extract the NET PAY amount from this payslip OCR text.

CRITICAL RULES:
1. Extract ONLY the NET PAY value - this is the take-home pay after all deductions
2. Look for keywords: "NET PAY", "NET", "TAKE HOME", "PAYABLE", "NET AMOUNT"
3. Return the exact number you find - no calculations
4. If you cannot find the net pay clearly, return 0
5. Return ONLY valid JSON, no other text

PAYSLIP TEXT:
{ocr_text}

Return ONLY this JSON format:
{{
  "net_pay": number,
  "currency": "ZMW or detected currency",
  "pay_period": "month/year if found or null",
  "employee_name": "name if found or null",
  "confidence": number between 0 and 1
}}

Extract the data and return JSON:"""

        # Use unified LLM call (supports Ollama or Groq)
        llm_text = call_llm(prompt)
        
        if llm_text is None:
            provider = LLM_PROVIDER.lower()
            if provider == 'groq':
                result['errors'].append("Groq API failed. Check GROQ_API_KEY environment variable.")
            else:
                result['errors'].append(f"Cannot connect to LLM at {OLLAMA_ENDPOINT}. Is Ollama running?")
            return result
        
        # Step 4: Parse JSON response
        payslip_data = parse_llm_response(llm_text)
        
        if payslip_data is None:
            result['errors'].append("Failed to parse LLM response as JSON")
            result['data'] = {'raw_response': llm_text[:500], 'ocr_text': ocr_text[:1000]}
            return result
        
        # Step 5: Validate and format result
        result['success'] = True
        result['data'] = payslip_data
        result['confidence'] = payslip_data.get('confidence', 0.5)
        
        # Format net_pay for display
        if 'net_pay' in payslip_data and payslip_data['net_pay']:
            net_pay = payslip_data['net_pay']
            if isinstance(net_pay, (int, float)):
                result['data']['net_pay_formatted'] = f"{net_pay:,.2f}"
        
    except Exception as e:
        result['errors'].append(f"Extraction error: {str(e)}")
    
    finally:
        result['processing_time_seconds'] = time.time() - start_time
    
    return result



def parse_llm_response(llm_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response with robust JSON cleaning
    """
    try:
        # Remove markdown formatting
        if '```json' in llm_text:
            llm_text = llm_text.split('```json')[1].split('```')[0].strip()
        elif '```' in llm_text:
            llm_text = llm_text.split('```')[1].split('```')[0].strip()
        
        # Extract JSON object
        json_start = llm_text.find('{')
        json_end = llm_text.rfind('}')
        if json_start >= 0 and json_end > json_start:
            llm_text = llm_text[json_start:json_end + 1]
        
        # Fix common JSON issues
        llm_text = llm_text.replace('True', 'true')
        llm_text = llm_text.replace('False', 'false')
        llm_text = llm_text.replace('None', 'null')
        
        # Remove trailing commas
        llm_text = re.sub(r',(\s*[}\]])', r'\1', llm_text)
        
        # Remove comments
        llm_text = re.sub(r'//[^\n]*', '', llm_text)
        
        # Parse
        return json.loads(llm_text)
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Attempted to parse: {llm_text[:200]}...")
        return None


def extract_netpay_from_file(file_path: str, save_json: bool = False) -> Dict[str, Any]:
    """
    Extract net pay from payslip file and optionally save result
    
    Args:
        file_path: Path to payslip image or PDF
        save_json: Whether to save result to JSON file
    
    Returns:
        Extraction result dictionary
    """
    print(f"Processing: {file_path}")
    print("=" * 60)
    
    result = extract_payslip_llm(file_path=file_path)
    
    # Display results
    print(f"\nStatus: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Processing time: {result['processing_time_seconds']:.2f}s")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if result['errors']:
        print(f"Errors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['success']:
        data = result['data']
        print(f"\nExtracted Data:")
        print(f"  Net Pay: {data.get('net_pay_formatted', data.get('net_pay', 'N/A'))} {data.get('currency', '')}")
        print(f"  Employee: {data.get('employee_name', 'N/A')}")
        print(f"  Pay Period: {data.get('pay_period', 'N/A')}")
        
        if save_json:
            output_path = Path(file_path).stem + '_extracted.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to: {output_path}")
    
    print("=" * 60)
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        file_path = sys.argv[1]
        result = extract_netpay_from_file(file_path, save_json=True)
    else:
        print("PAYSLIP NET PAY EXTRACTION - LLM BASED")
        print("=" * 60)
        print(f"Model: {LLM_MODEL}")
        print(f"Endpoint: {LLM_ENDPOINT}")
        print()
        print("Usage:")
        print("  Command Line:")
        print("    python extract_payslip_llm.py payslip.pdf")
        print()
        print("  Python API:")
        print("    from extract_payslip_llm import extract_payslip_llm")
        print("    result = extract_payslip_llm(file_path='payslip.pdf')")
        print("    print(result['data']['net_pay'])")
        print("=" * 60)
