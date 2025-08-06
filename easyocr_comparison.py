import cv2
import numpy as np
import pytesseract
import easyocr
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class OCRComparisonSystem:
    def __init__(self, ground_truth_file="ground_truth.json"):
        """OCR karşılaştırma sistemi başlatıcı"""
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.results = []
        
        # EasyOCR reader'ı başlat (İngilizce)
        self.easyocr_reader = easyocr.Reader(['en'])
        
        # Ön işleme yöntemleri
        self.preprocessing_methods = {
            'original': self.no_preprocessing,
            'grayscale': self.grayscale_preprocessing,
            'noise_reduction': self.noise_reduction,
            'contrast_enhancement': self.contrast_enhancement,
            'threshold': self.threshold_preprocessing,
            'adaptive_threshold': self.adaptive_threshold,
            'morphological': self.morphological_preprocessing,
            'combined': self.combined_preprocessing
        }
        
    def load_ground_truth(self, file_path):
        """Ground truth verilerini yükle"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)['ground_truth']
    
    def no_preprocessing(self, image):
        """Hiçbir ön işleme yapmadan orijinal görüntüyü döndür"""
        return image
    
    def grayscale_preprocessing(self, image):
        """Gri tonlama ön işleme"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def noise_reduction(self, image):
        """Gürültü azaltma"""
        return cv2.medianBlur(image, 3)
    
    def contrast_enhancement(self, image):
        """Kontrast artırma"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def threshold_preprocessing(self, image):
        """Basit threshold uygulama"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def adaptive_threshold(self, image):
        """Adaptif threshold uygulama"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    def morphological_preprocessing(self, image):
        """Morfolojik işlemler"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1,1), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def combined_preprocessing(self, image):
        """Birleşik ön işleme yöntemi"""
        # Gri tonlama
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        image = cv2.medianBlur(image, 3)
        
        # Kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # Adaptif threshold
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morfolojik işlemler
        kernel = np.ones((1,1), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image
    
    def extract_text_tesseract(self, image, method='combined'):
        """Tesseract ile metin çıkarma"""
        # Ön işleme uygula
        processed_image = self.preprocessing_methods[method](image)
        
        # Tesseract konfigürasyonu
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().:/- '
        
        # OCR uygula
        data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Sonuçları işle
        extracted_texts = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Güven skoru 30'dan büyük olanları al
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    conf = data['conf'][i]
                    extracted_texts.append({
                        'text': text,
                        'bbox': (x, y, x + w, y + h),
                        'confidence': conf
                    })
        
        return extracted_texts, processed_image
    
    def extract_text_easyocr(self, image, method='combined'):
        """EasyOCR ile metin çıkarma"""
        # Ön işleme uygula
        processed_image = self.preprocessing_methods[method](image)
        
        # EasyOCR ile OCR uygula
        results = self.easyocr_reader.readtext(processed_image)
        
        # Sonuçları işle
        extracted_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Güven skoru 0.3'ten büyük olanları al
                # Bbox formatını dönüştür
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                extracted_texts.append({
                    'text': text.strip(),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence * 100  # Yüzdeye çevir
                })
        
        return extracted_texts, processed_image
    
    def calculate_similarity(self, text1, text2):
        """İki metin arasındaki benzerliği hesapla"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_best_match(self, extracted_text, ground_truth_values):
        """Ground truth değerleri arasında en iyi eşleşmeyi bul"""
        best_match = None
        best_similarity = 0
        
        for key, value in ground_truth_values.items():
            if value:  # Boş değerleri atla
                similarity = self.calculate_similarity(extracted_text, str(value))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (key, value, similarity)
        
        return best_match
    
    def process_image(self, image_path, method='combined', ocr_type='tesseract'):
        """Tek bir görüntüyü işle"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi")
            return None
        
        # OCR uygula
        if ocr_type == 'tesseract':
            extracted_texts, processed_image = self.extract_text_tesseract(image, method)
        else:  # easyocr
            extracted_texts, processed_image = self.extract_text_easyocr(image, method)
        
        # Ground truth ile karşılaştır
        matches = []
        for extracted in extracted_texts:
            best_match = self.find_best_match(extracted['text'], self.ground_truth)
            if best_match and best_match[2] > 0.3:  # %30'dan fazla benzerlik
                matches.append({
                    'extracted_text': extracted['text'],
                    'ground_truth_key': best_match[0],
                    'ground_truth_value': best_match[1],
                    'similarity': best_match[2],
                    'confidence': extracted['confidence'],
                    'bbox': extracted['bbox']
                })
        
        return {
            'image_path': image_path,
            'method': method,
            'ocr_type': ocr_type,
            'extracted_texts': extracted_texts,
            'matches': matches,
            'processed_image': processed_image,
            'original_image': image
        }
    
    def create_visualization(self, result):
        """Görsel çıktı oluştur"""
        if not result:
            return None
        
        # Orijinal görüntüyü kopyala
        vis_image = result['original_image'].copy()
        
        # OCR tipini belirle
        ocr_type = result['ocr_type']
        color_map = {
            'tesseract': (0, 255, 0),  # Yeşil
            'easyocr': (255, 0, 0)      # Mavi
        }
        box_color = color_map.get(ocr_type, (0, 255, 0))
        
        # Tüm okunan metinler için bounding box'ları çiz
        for extracted in result['extracted_texts']:
            x1, y1, x2, y2 = extracted['bbox']
            
            # Tüm metinler için box çiz
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, 1)
            
            # Metin ekle
            text = f"{extracted['text']}"
            cv2.putText(vis_image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
        
        # Ground truth ile eşleşen metinler için kalın box'ları çiz
        for match in result['matches']:
            x1, y1, x2, y2 = match['bbox']
            
            # Eşleşen metinler için kalın box çiz
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Sarı
            
            # Metin ekle
            text = f"{match['ground_truth_key']}: {match['extracted_text']}"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Benzerlik skorunu da ekle
            similarity_text = f"Benzerlik: {match['similarity']:.2f}"
            cv2.putText(vis_image, similarity_text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return vis_image
    
    def create_comparison_visualization(self, results_for_image):
        """Aynı görüntü için farklı OCR'ların karşılaştırmalı görselleştirmesi"""
        if not results_for_image:
            return None
        
        # İlk sonucun orijinal görüntüsünü al
        original_image = results_for_image[0]['original_image']
        height, width = original_image.shape[:2]
        
        # OCR sayısını hesapla
        num_ocrs = len(results_for_image)
        cols = min(2, num_ocrs)  # Maksimum 2 sütun (Tesseract vs EasyOCR)
        rows = (num_ocrs + cols - 1) // cols
        
        # Büyük bir görüntü oluştur
        comparison_image = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
        
        for idx, result in enumerate(results_for_image):
            row = idx // cols
            col = idx % cols
            
            # Görselleştirme oluştur
            vis_image = self.create_visualization(result)
            if vis_image is not None:
                # OCR adını ekle
                ocr_name = result['ocr_type'].upper()
                method_name = result['method'].upper()
                cv2.putText(vis_image, f"{ocr_name} - {method_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # İstatistikleri ekle
                stats_text = f"Metin: {len(result['extracted_texts'])} | Eşleşme: {len(result['matches'])}"
                cv2.putText(vis_image, stats_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Görüntüyü yerleştir
                y_start = row * height
                y_end = (row + 1) * height
                x_start = col * width
                x_end = (col + 1) * width
                comparison_image[y_start:y_end, x_start:x_end] = vis_image
        
        return comparison_image
    
    def process_all_images(self, images_dir="images"):
        """Tüm görüntüleri işle"""
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Her görüntü için tüm OCR'ların sonuçlarını sakla
        image_results = {}
        
        # OCR tipleri
        ocr_types = ['tesseract', 'easyocr']
        
        for ocr_type in ocr_types:
            print(f"\n{ocr_type.upper()} OCR ile işleniyor...")
            
            for method_name in self.preprocessing_methods.keys():
                print(f"  {method_name.upper()} yöntemi...")
                
                for image_file in image_files:
                    image_path = os.path.join(images_dir, image_file)
                    result = self.process_image(image_path, method_name, ocr_type)
                    
                    if result:
                        # Görsel çıktı oluştur
                        vis_image = self.create_visualization(result)
                        
                        # Sonuçları kaydet
                        self.results.append(result)
                        
                        # Görüntü bazında sonuçları sakla
                        if image_file not in image_results:
                            image_results[image_file] = []
                        image_results[image_file].append(result)
                        
                        # Görsel çıktıyı kaydet
                        if vis_image is not None:
                            output_dir = f"outputs_comparison/{ocr_type}/{method_name}"
                            os.makedirs(output_dir, exist_ok=True)
                            
                            vis_path = os.path.join(output_dir, f"vis_{image_file}")
                            cv2.imwrite(vis_path, vis_image)
                            
                            # İşlenmiş görüntüyü de kaydet
                            processed_path = os.path.join(output_dir, f"processed_{image_file}")
                            cv2.imwrite(processed_path, result['processed_image'])
                            
                            # İşlenmiş görüntüde de bounding box'ları göster
                            processed_with_boxes = result['processed_image'].copy()
                            if len(processed_with_boxes.shape) == 2:  # Gri tonlama ise
                                processed_with_boxes = cv2.cvtColor(processed_with_boxes, cv2.COLOR_GRAY2BGR)
                            
                            # Tüm okunan metinler için bounding box'ları çiz
                            for extracted in result['extracted_texts']:
                                x1, y1, x2, y2 = extracted['bbox']
                                cv2.rectangle(processed_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            processed_vis_path = os.path.join(output_dir, f"processed_with_boxes_{image_file}")
                            cv2.imwrite(processed_vis_path, processed_with_boxes)
        
        # Karşılaştırmalı görselleştirmeler oluştur
        print("\nKarşılaştırmalı görselleştirmeler oluşturuluyor...")
        comparison_dir = "outputs_comparison/comparisons"
        os.makedirs(comparison_dir, exist_ok=True)
        
        for image_file, results_for_image in image_results.items():
            if len(results_for_image) > 1:  # En az 2 OCR varsa
                comparison_image = self.create_comparison_visualization(results_for_image)
                if comparison_image is not None:
                    comparison_path = os.path.join(comparison_dir, f"comparison_{image_file}")
                    cv2.imwrite(comparison_path, comparison_image)
                    print(f"  Karşılaştırma kaydedildi: {comparison_path}")
        
        return self.results
    
    def generate_reports(self):
        """Raporlar oluştur"""
        # Sonuçları analiz et
        analysis = {}
        
        for result in self.results:
            method = result['method']
            ocr_type = result['ocr_type']
            key = f"{ocr_type}_{method}"
            
            if key not in analysis:
                analysis[key] = {
                    'ocr_type': ocr_type,
                    'method': method,
                    'total_matches': 0,
                    'total_similarity': 0,
                    'total_confidence': 0,
                    'image_count': 0,
                    'matches_by_field': {}
                }
            
            analysis[key]['image_count'] += 1
            analysis[key]['total_matches'] += len(result['matches'])
            
            for match in result['matches']:
                analysis[key]['total_similarity'] += match['similarity']
                analysis[key]['total_confidence'] += match['confidence']
                
                field = match['ground_truth_key']
                if field not in analysis[key]['matches_by_field']:
                    analysis[key]['matches_by_field'][field] = {
                        'count': 0,
                        'avg_similarity': 0,
                        'avg_confidence': 0
                    }
                
                analysis[key]['matches_by_field'][field]['count'] += 1
                analysis[key]['matches_by_field'][field]['avg_similarity'] += match['similarity']
                analysis[key]['matches_by_field'][field]['avg_confidence'] += match['confidence']
        
        # Ortalamaları hesapla
        for key in analysis:
            if analysis[key]['total_matches'] > 0:
                analysis[key]['avg_similarity'] = analysis[key]['total_similarity'] / analysis[key]['total_matches']
                analysis[key]['avg_confidence'] = analysis[key]['total_confidence'] / analysis[key]['total_matches']
            
            for field in analysis[key]['matches_by_field']:
                count = analysis[key]['matches_by_field'][field]['count']
                if count > 0:
                    analysis[key]['matches_by_field'][field]['avg_similarity'] /= count
                    analysis[key]['matches_by_field'][field]['avg_confidence'] /= count
        
        return analysis
    
    def create_performance_charts(self, analysis):
        """Performans grafikleri oluştur"""
        # Grafikler için veri hazırla
        keys = list(analysis.keys())
        avg_similarities = [analysis[k]['avg_similarity'] for k in keys]
        avg_confidences = [analysis[k]['avg_confidence'] for k in keys]
        total_matches = [analysis[k]['total_matches'] for k in keys]
        
        # OCR tiplerini ayır
        tesseract_keys = [k for k in keys if 'tesseract' in k]
        easyocr_keys = [k for k in keys if 'easyocr' in k]
        
        # Grafikleri oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Ortalama benzerlik skorları
        x_pos = np.arange(len(keys))
        colors = ['green' if 'tesseract' in k else 'blue' for k in keys]
        ax1.bar(x_pos, avg_similarities, color=colors)
        ax1.set_title('Ortalama Benzerlik Skorları (Tesseract vs EasyOCR)')
        ax1.set_ylabel('Benzerlik Skoru')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(keys, rotation=45, ha='right')
        
        # Ortalama güven skorları
        ax2.bar(x_pos, avg_confidences, color=colors)
        ax2.set_title('Ortalama Güven Skorları (Tesseract vs EasyOCR)')
        ax2.set_ylabel('Güven Skoru')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(keys, rotation=45, ha='right')
        
        # Toplam eşleşme sayıları
        ax3.bar(x_pos, total_matches, color=colors)
        ax3.set_title('Toplam Eşleşme Sayıları (Tesseract vs EasyOCR)')
        ax3.set_ylabel('Eşleşme Sayısı')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(keys, rotation=45, ha='right')
        
        # OCR tipi bazında karşılaştırma
        tesseract_avg = np.mean([analysis[k]['avg_similarity'] for k in tesseract_keys]) if tesseract_keys else 0
        easyocr_avg = np.mean([analysis[k]['avg_similarity'] for k in easyocr_keys]) if easyocr_keys else 0
        
        ocr_comparison = ['Tesseract', 'EasyOCR']
        ocr_scores = [tesseract_avg, easyocr_avg]
        colors_ocr = ['green', 'blue']
        
        ax4.bar(ocr_comparison, ocr_scores, color=colors_ocr)
        ax4.set_title('OCR Tipi Bazında Ortalama Benzerlik')
        ax4.set_ylabel('Ortalama Benzerlik Skoru')
        
        plt.tight_layout()
        plt.savefig('ocr_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_detailed_report(self, analysis):
        """Detaylı rapor kaydet"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_fields': list(self.ground_truth.keys()),
            'analysis': analysis,
            'summary': {
                'best_tesseract': max([k for k in analysis.keys() if 'tesseract' in k], key=lambda x: analysis[x]['avg_similarity']) if any('tesseract' in k for k in analysis.keys()) else None,
                'best_easyocr': max([k for k in analysis.keys() if 'easyocr' in k], key=lambda x: analysis[x]['avg_similarity']) if any('easyocr' in k for k in analysis.keys()) else None,
                'total_images_processed': sum(analysis[k]['image_count'] for k in analysis),
                'total_matches_found': sum(analysis[k]['total_matches'] for k in analysis)
            }
        }
        
        with open('ocr_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # CSV raporu da oluştur
        report_data = []
        for key, data in analysis.items():
            for field, field_data in data['matches_by_field'].items():
                report_data.append({
                    'OCR_Type': data['ocr_type'],
                    'Method': data['method'],
                    'Field': field,
                    'Match_Count': field_data['count'],
                    'Avg_Similarity': field_data['avg_similarity'],
                    'Avg_Confidence': field_data['avg_confidence']
                })
        
        df = pd.DataFrame(report_data)
        df.to_csv('ocr_comparison_report.csv', index=False)
        
        return report

def main():
    """Ana fonksiyon"""
    print("OCR Karşılaştırma Sistemi Başlatılıyor...")
    
    # OCR karşılaştırma sistemi oluştur
    ocr_comparison = OCRComparisonSystem()
    
    # Tüm görüntüleri işle
    print("Görüntüler işleniyor...")
    results = ocr_comparison.process_all_images()
    
    # Analiz yap
    print("Analiz yapılıyor...")
    analysis = ocr_comparison.generate_reports()
    
    # Grafikleri oluştur
    print("Grafikler oluşturuluyor...")
    ocr_comparison.create_performance_charts(analysis)
    
    # Detaylı rapor kaydet
    print("Raporlar kaydediliyor...")
    report = ocr_comparison.save_detailed_report(analysis)
    
    print(f"\n{'='*60}")
    print(f"OCR KARŞILAŞTIRMA TAMAMLANDI!")
    print(f"{'='*60}")
    
    if report['summary']['best_tesseract']:
        print(f"🏆 En iyi Tesseract: {report['summary']['best_tesseract']}")
    if report['summary']['best_easyocr']:
        print(f"🏆 En iyi EasyOCR: {report['summary']['best_easyocr']}")
    
    print(f"📸 İşlenen görüntü sayısı: {report['summary']['total_images_processed']}")
    print(f"✅ Bulunan eşleşme sayısı: {report['summary']['total_matches_found']}")
    
    print(f"\n📁 ÇIKTILAR:")
    print(f"  📂 outputs_comparison/ klasörü:")
    print(f"    - Her OCR tipi için ayrı klasör")
    print(f"    - Her yöntem için ayrı klasör")
    print(f"    - vis_[görüntü].png: Bounding box'lı görüntüler")
    print(f"    - processed_[görüntü].png: İşlenmiş görüntüler")
    print(f"    - processed_with_boxes_[görüntü].png: İşlenmiş görüntüler + box'lar")
    print(f"  📂 outputs_comparison/comparisons/ klasörü:")
    print(f"    - comparison_[görüntü].png: Tesseract vs EasyOCR karşılaştırması")
    print(f"  📊 Raporlar:")
    print(f"    - ocr_comparison_analysis.png (karşılaştırma grafikleri)")
    print(f"    - ocr_comparison_report.json (detaylı JSON raporu)")
    print(f"    - ocr_comparison_report.csv (CSV raporu)")
    
    print(f"\n🎨 GÖRSEL ÇIKTILAR:")
    print(f"  🟢 Yeşil box'lar: Tesseract sonuçları")
    print(f"  🔵 Mavi box'lar: EasyOCR sonuçları")
    print(f"  🟡 Sarı box'lar: Ground truth ile eşleşen metinler")

if __name__ == "__main__":
    main() 