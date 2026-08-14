"""
Phase 5: OCR Metrics Module

Provides OCR-based evaluation metrics for document enhancement assessment.
Integrates with Tesseract OCR engine to evaluate text readability improvements.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import os

def compute_ocr_metrics(image: np.ndarray, lang: str = 'eng') -> Dict:
    """
    Compute OCR metrics for a single image.
    """
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
    except ImportError:
        print("Warning: pytesseract not installed.")
        return {'confidence': 0.0, 'word_count': 0, 'available': False}
    except Exception as e:
        print(f"Warning: Tesseract configuration error: {e}")
        return {'confidence': 0.0, 'word_count': 0, 'available': False}
    
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    data = pytesseract.image_to_data(image, lang=lang, 
                                      output_type=pytesseract.Output.DICT)
    
    confidences = [c for c in data['conf'] if c > -1]
    avg_confidence = np.mean(confidences) if confidences else 0.0
    word_count = len([w for w in data['text'] if w.strip()])
    
    return {
        'confidence': avg_confidence,
        'word_count': word_count,
        'available': True
    }

def compare_readability(degraded: np.ndarray, enhanced: np.ndarray,
                        reference: Optional[np.ndarray] = None,
                        lang: str = 'eng') -> Dict:
    """
    Compare readability between degraded and enhanced images.
    """
    degraded_result = compute_ocr_metrics(degraded, lang)
    enhanced_result = compute_ocr_metrics(enhanced, lang)
    
    results = {
        'degraded_confidence': degraded_result['confidence'],
        'enhanced_confidence': enhanced_result['confidence'],
        'improvement': enhanced_result['confidence'] - degraded_result['confidence'],
        'degraded_words': degraded_result['word_count'],
        'enhanced_words': enhanced_result['word_count'],
    }
    
    if reference is not None:
        ref_result = compute_ocr_metrics(reference, lang)
        results['reference_confidence'] = ref_result['confidence']
        results['reference_words'] = ref_result['word_count']
    
    return results