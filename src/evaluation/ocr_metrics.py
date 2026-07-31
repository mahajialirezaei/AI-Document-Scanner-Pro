"""
Phase 5: OCR Metrics Module

Provides OCR-based evaluation metrics for document enhancement assessment.
Integrates with Tesseract OCR engine to evaluate text readability improvements.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def compute_ocr_metrics(image: np.ndarray, lang: str = 'eng') -> Dict:
    """
    Compute OCR metrics for a single image.
    
    Args:
        image: Input image (H, W, C) in range [0, 255] or [0, 1]
        lang: Tesseract language code
        
    Returns:
        Dictionary with OCR confidence and word count
    """
    try:
        import pytesseract
    except ImportError:
        print("Warning: pytesseract not installed.")
        return {'confidence': 0.0, 'word_count': 0, 'available': False}
    
    # Ensure proper format
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    # Get OCR data
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
    
    Args:
        degraded: Degraded input image
        enhanced: Enhanced output image
        reference: Optional clean reference scan
        lang: Tesseract language code
        
    Returns:
        Dictionary with comparative metrics
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