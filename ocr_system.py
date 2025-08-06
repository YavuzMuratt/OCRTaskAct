import cv2
import numpy as np
import pytesseract
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

class OCRSystem:
    def __init__(self, ground_truth_file="ground_truth.json"):
        """OCR sistemi başlatıcı"""
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.results = []
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
    
    def extract_text_with_boxes(self, image, method='combined'):
        """OCR ile metin çıkarma ve bounding box'ları alma"""
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
    
    def process_image(self, image_path, method='combined'):
        """Tek bir görüntüyü işle"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi")
            return None
        
        # OCR uygula
        extracted_texts, processed_image = self.extract_text_with_boxes(image, method)
        
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
        
        # Tüm okunan metinler için bounding box'ları çiz (kırmızı)
        for extracted in result['extracted_texts']:
            x1, y1, x2, y2 = extracted['bbox']
            
            # Tüm metinler için kırmızı box çiz
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            # Metin ekle (küçük font)
            text = f"{extracted['text']}"
            cv2.putText(vis_image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Ground truth ile eşleşen metinler için yeşil box'ları çiz
        for match in result['matches']:
            x1, y1, x2, y2 = match['bbox']
            
            # Eşleşen metinler için yeşil box çiz (kalın)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Metin ekle (büyük font)
            text = f"{match['ground_truth_key']}: {match['extracted_text']}"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Benzerlik skorunu da ekle
            similarity_text = f"Benzerlik: {match['similarity']:.2f}"
            cv2.putText(vis_image, similarity_text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis_image
    
    def create_comparison_visualization(self, results_for_image):
        """Aynı görüntü için farklı yöntemlerin karşılaştırmalı görselleştirmesi"""
        if not results_for_image:
            return None
        
        # İlk sonucun orijinal görüntüsünü al
        original_image = results_for_image[0]['original_image']
        height, width = original_image.shape[:2]
        
        # Yöntem sayısını hesapla
        num_methods = len(results_for_image)
        cols = min(3, num_methods)  # Maksimum 3 sütun
        rows = (num_methods + cols - 1) // cols
        
        # Büyük bir görüntü oluştur
        comparison_image = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
        
        for idx, result in enumerate(results_for_image):
            row = idx // cols
            col = idx % cols
            
            # Görselleştirme oluştur
            vis_image = self.create_visualization(result)
            if vis_image is not None:
                # Yöntem adını ekle
                method_name = result['method'].upper()
                cv2.putText(vis_image, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
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
        
        # Her görüntü için tüm yöntemlerin sonuçlarını sakla
        image_results = {}
        
        for method_name in self.preprocessing_methods.keys():
            print(f"\n{method_name.upper()} yöntemi ile işleniyor...")
            
            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                result = self.process_image(image_path, method_name)
                
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
                        output_dir = f"outputs/{method_name}"
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
        comparison_dir = "outputs/comparisons"
        os.makedirs(comparison_dir, exist_ok=True)
        
        for image_file, results_for_image in image_results.items():
            if len(results_for_image) > 1:  # En az 2 yöntem varsa
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
            if method not in analysis:
                analysis[method] = {
                    'total_matches': 0,
                    'total_similarity': 0,
                    'total_confidence': 0,
                    'image_count': 0,
                    'matches_by_field': {}
                }
            
            analysis[method]['image_count'] += 1
            analysis[method]['total_matches'] += len(result['matches'])
            
            for match in result['matches']:
                analysis[method]['total_similarity'] += match['similarity']
                analysis[method]['total_confidence'] += match['confidence']
                
                field = match['ground_truth_key']
                if field not in analysis[method]['matches_by_field']:
                    analysis[method]['matches_by_field'][field] = {
                        'count': 0,
                        'avg_similarity': 0,
                        'avg_confidence': 0
                    }
                
                analysis[method]['matches_by_field'][field]['count'] += 1
                analysis[method]['matches_by_field'][field]['avg_similarity'] += match['similarity']
                analysis[method]['matches_by_field'][field]['avg_confidence'] += match['confidence']
        
        # Ortalamaları hesapla
        for method in analysis:
            if analysis[method]['total_matches'] > 0:
                analysis[method]['avg_similarity'] = analysis[method]['total_similarity'] / analysis[method]['total_matches']
                analysis[method]['avg_confidence'] = analysis[method]['total_confidence'] / analysis[method]['total_matches']
            
            for field in analysis[method]['matches_by_field']:
                count = analysis[method]['matches_by_field'][field]['count']
                if count > 0:
                    analysis[method]['matches_by_field'][field]['avg_similarity'] /= count
                    analysis[method]['matches_by_field'][field]['avg_confidence'] /= count
        
        return analysis
    
    def create_performance_charts(self, analysis):
        """Performans grafikleri oluştur"""
        # Grafikler için veri hazırla
        methods = list(analysis.keys())
        avg_similarities = [analysis[m]['avg_similarity'] for m in methods]
        avg_confidences = [analysis[m]['avg_confidence'] for m in methods]
        total_matches = [analysis[m]['total_matches'] for m in methods]
        
        # Grafikleri oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Ortalama benzerlik skorları
        ax1.bar(methods, avg_similarities, color='skyblue')
        ax1.set_title('Ortalama Benzerlik Skorları')
        ax1.set_ylabel('Benzerlik Skoru')
        ax1.tick_params(axis='x', rotation=45)
        
        # Ortalama güven skorları
        ax2.bar(methods, avg_confidences, color='lightgreen')
        ax2.set_title('Ortalama Güven Skorları')
        ax2.set_ylabel('Güven Skoru')
        ax2.tick_params(axis='x', rotation=45)
        
        # Toplam eşleşme sayıları
        ax3.bar(methods, total_matches, color='salmon')
        ax3.set_title('Toplam Eşleşme Sayıları')
        ax3.set_ylabel('Eşleşme Sayısı')
        ax3.tick_params(axis='x', rotation=45)
        
        # En iyi yöntem seçimi (benzerlik + güven)
        combined_scores = [(s + c) / 2 for s, c in zip(avg_similarities, avg_confidences)]
        ax4.bar(methods, combined_scores, color='gold')
        ax4.set_title('Birleşik Performans Skorları')
        ax4.set_ylabel('Birleşik Skor')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_detailed_report(self, analysis):
        """Detaylı rapor kaydet"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_fields': list(self.ground_truth.keys()),
            'analysis': analysis,
            'summary': {
                'best_method': max(analysis.keys(), key=lambda x: analysis[x]['avg_similarity']),
                'total_images_processed': sum(analysis[m]['image_count'] for m in analysis),
                'total_matches_found': sum(analysis[m]['total_matches'] for m in analysis)
            }
        }
        
        with open('ocr_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # CSV raporu da oluştur
        report_data = []
        for method, data in analysis.items():
            for field, field_data in data['matches_by_field'].items():
                report_data.append({
                    'Method': method,
                    'Field': field,
                    'Match_Count': field_data['count'],
                    'Avg_Similarity': field_data['avg_similarity'],
                    'Avg_Confidence': field_data['avg_confidence']
                })
        
        df = pd.DataFrame(report_data)
        df.to_csv('ocr_report.csv', index=False)
        
        return report

def main():
    """Ana fonksiyon"""
    print("OCR Sistemi Başlatılıyor...")
    
    # OCR sistemi oluştur
    ocr_system = OCRSystem()
    
    # Tüm görüntüleri işle
    print("Görüntüler işleniyor...")
    results = ocr_system.process_all_images()
    
    # Analiz yap
    print("Analiz yapılıyor...")
    analysis = ocr_system.generate_reports()
    
    # Grafikleri oluştur
    print("Grafikler oluşturuluyor...")
    ocr_system.create_performance_charts(analysis)
    
    # Detaylı rapor kaydet
    print("Raporlar kaydediliyor...")
    report = ocr_system.save_detailed_report(analysis)
    
    print(f"\n{'='*60}")
    print(f"İŞLEM TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"🏆 En iyi yöntem: {report['summary']['best_method']}")
    print(f"📸 İşlenen görüntü sayısı: {report['summary']['total_images_processed']}")
    print(f"✅ Bulunan eşleşme sayısı: {report['summary']['total_matches_found']}")
    
    print(f"\n📁 ÇIKTILAR:")
    print(f"  📂 outputs/ klasörü:")
    print(f"    - Her yöntem için ayrı klasör")
    print(f"    - vis_[görüntü].png: Bounding box'lı görüntüler")
    print(f"    - processed_[görüntü].png: İşlenmiş görüntüler")
    print(f"    - processed_with_boxes_[görüntü].png: İşlenmiş görüntüler + box'lar")
    print(f"  📂 outputs/comparisons/ klasörü:")
    print(f"    - comparison_[görüntü].png: Tüm yöntemlerin karşılaştırması")
    print(f"  📊 Raporlar:")
    print(f"    - performance_analysis.png (performans grafikleri)")
    print(f"    - ocr_report.json (detaylı JSON raporu)")
    print(f"    - ocr_report.csv (CSV raporu)")
    
    print(f"\n🎨 GÖRSEL ÇIKTILAR:")
    print(f"  🔴 Kırmızı box'lar: Tüm okunan metinler")
    print(f"  🟢 Yeşil box'lar: Ground truth ile eşleşen metinler")
    print(f"  📊 Her görüntüde yöntem adı ve istatistikler gösterilir")

if __name__ == "__main__":
    main() 