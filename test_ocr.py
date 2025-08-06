#!/usr/bin/env python3
"""
OCR Sistemi Test Scripti
Bu script OCR sistemini test etmek ve sonuçları görüntülemek için kullanılır.
"""

import os
import sys
import json
from ocr_system import OCRSystem

def test_single_image():
    """Tek bir görüntüyü test et"""
    print("Tek görüntü testi başlatılıyor...")
    
    # OCR sistemi oluştur
    ocr = OCRSystem()
    
    # İlk görüntüyü al
    images_dir = "images"
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Hata: images klasöründe görüntü bulunamadı!")
        return
    
    test_image = os.path.join(images_dir, image_files[0])
    print(f"Test edilen görüntü: {test_image}")
    
    # Farklı yöntemlerle test et
    methods = ['original', 'combined', 'adaptive_threshold']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"{method.upper()} YÖNTEMİ")
        print(f"{'='*50}")
        
        result = ocr.process_image(test_image, method)
        
        if result:
            print(f"📊 İSTATİSTİKLER:")
            print(f"  - Bulunan metin sayısı: {len(result['extracted_texts'])}")
            print(f"  - Eşleşme sayısı: {len(result['matches'])}")
            
            print(f"\n📝 TÜM OKUNAN METİNLER:")
            for i, extracted in enumerate(result['extracted_texts']):
                print(f"  {i+1}. '{extracted['text']}' (Güven: {extracted['confidence']:.1f}%, BBox: {extracted['bbox']})")
            
            print(f"\n✅ GROUND TRUTH İLE EŞLEŞENLER:")
            for match in result['matches']:
                print(f"  - {match['ground_truth_key']}: '{match['extracted_text']}'")
                print(f"    Ground Truth: '{match['ground_truth_value']}'")
                print(f"    Benzerlik: {match['similarity']:.2f} | Güven: {match['confidence']:.1f}%")
                print(f"    Bounding Box: {match['bbox']}")
            
            # Görsel çıktı oluştur
            vis_image = ocr.create_visualization(result)
            if vis_image is not None:
                output_dir = f"outputs/{method}"
                os.makedirs(output_dir, exist_ok=True)
                
                vis_path = os.path.join(output_dir, f"vis_{os.path.basename(test_image)}")
                cv2.imwrite(vis_path, vis_image)
                print(f"\n🖼️  Görsel çıktı kaydedildi: {vis_path}")
                
                # İşlenmiş görüntüyü de kaydet
                processed_path = os.path.join(output_dir, f"processed_{os.path.basename(test_image)}")
                cv2.imwrite(processed_path, result['processed_image'])
                print(f"🖼️  İşlenmiş görüntü kaydedildi: {processed_path}")
            
            # JSON formatında detaylı sonuç kaydet
            result_json = {
                'image_path': result['image_path'],
                'method': result['method'],
                'extracted_texts': result['extracted_texts'],
                'matches': result['matches'],
                'statistics': {
                    'total_texts': len(result['extracted_texts']),
                    'total_matches': len(result['matches']),
                    'avg_confidence': sum(t['confidence'] for t in result['extracted_texts']) / len(result['extracted_texts']) if result['extracted_texts'] else 0,
                    'avg_similarity': sum(m['similarity'] for m in result['matches']) / len(result['matches']) if result['matches'] else 0
                }
            }
            
            json_path = os.path.join(output_dir, f"results_{os.path.basename(test_image)}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"📄 Detaylı sonuçlar kaydedildi: {json_path}")
            
        else:
            print("❌ Görüntü işlenemedi!")

def quick_test():
    """Hızlı test - sadece birkaç yöntemle"""
    print("Hızlı OCR testi başlatılıyor...")
    
    ocr = OCRSystem()
    
    # Sadece birkaç yöntemle test et
    test_methods = ['original', 'combined']
    
    for method in test_methods:
        print(f"\n{'='*40}")
        print(f"{method.upper()} YÖNTEMİ")
        print(f"{'='*40}")
        
        # İlk 2 görüntüyü test et
        images_dir = "images"
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:2]
        
        total_matches = 0
        total_similarity = 0
        total_texts = 0
        total_confidence = 0
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            print(f"\n📸 İşlenen görüntü: {image_file}")
            
            result = ocr.process_image(image_path, method)
            
            if result:
                total_texts += len(result['extracted_texts'])
                total_confidence += sum(t['confidence'] for t in result['extracted_texts'])
                total_matches += len(result['matches'])
                
                for match in result['matches']:
                    total_similarity += match['similarity']
                
                print(f"  - Okunan metin sayısı: {len(result['extracted_texts'])}")
                print(f"  - Eşleşme sayısı: {len(result['matches'])}")
                
                # En iyi eşleşmeleri göster
                if result['matches']:
                    best_match = max(result['matches'], key=lambda x: x['similarity'])
                    print(f"  - En iyi eşleşme: {best_match['ground_truth_key']} = '{best_match['extracted_text']}' (Benzerlik: {best_match['similarity']:.2f})")
        
        if total_texts > 0:
            avg_confidence = total_confidence / total_texts
            print(f"\n📊 TOPLAM İSTATİSTİKLER:")
            print(f"  - Toplam okunan metin: {total_texts}")
            print(f"  - Toplam eşleşme: {total_matches}")
            print(f"  - Ortalama güven skoru: {avg_confidence:.1f}%")
            
            if total_matches > 0:
                avg_similarity = total_similarity / total_matches
                print(f"  - Ortalama benzerlik: {avg_similarity:.2f}")

def detailed_test():
    """Detaylı test - tüm yöntemlerle ve kapsamlı raporlama"""
    print("Detaylı OCR testi başlatılıyor...")
    
    ocr = OCRSystem()
    
    # Tüm yöntemlerle test et
    all_methods = list(ocr.preprocessing_methods.keys())
    
    # İlk 3 görüntüyü test et
    images_dir = "images"
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]
    
    results_summary = {}
    
    for method in all_methods:
        print(f"\n{'='*60}")
        print(f"{method.upper()} YÖNTEMİ")
        print(f"{'='*60}")
        
        method_stats = {
            'total_texts': 0,
            'total_matches': 0,
            'total_confidence': 0,
            'total_similarity': 0,
            'images_processed': 0
        }
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            print(f"\n📸 {image_file}")
            
            result = ocr.process_image(image_path, method)
            
            if result:
                method_stats['images_processed'] += 1
                method_stats['total_texts'] += len(result['extracted_texts'])
                method_stats['total_confidence'] += sum(t['confidence'] for t in result['extracted_texts'])
                method_stats['total_matches'] += len(result['matches'])
                
                for match in result['matches']:
                    method_stats['total_similarity'] += match['similarity']
                
                print(f"  ✅ Metinler: {len(result['extracted_texts'])} | Eşleşmeler: {len(result['matches'])}")
                
                # En iyi 3 eşleşmeyi göster
                if result['matches']:
                    top_matches = sorted(result['matches'], key=lambda x: x['similarity'], reverse=True)[:3]
                    for match in top_matches:
                        print(f"    - {match['ground_truth_key']}: '{match['extracted_text']}' ({match['similarity']:.2f})")
        
        # Ortalamaları hesapla
        if method_stats['total_texts'] > 0:
            method_stats['avg_confidence'] = method_stats['total_confidence'] / method_stats['total_texts']
        else:
            method_stats['avg_confidence'] = 0
            
        if method_stats['total_matches'] > 0:
            method_stats['avg_similarity'] = method_stats['total_similarity'] / method_stats['total_matches']
        else:
            method_stats['avg_similarity'] = 0
        
        results_summary[method] = method_stats
        
        print(f"\n📊 {method.upper()} ÖZET:")
        print(f"  - İşlenen görüntü: {method_stats['images_processed']}")
        print(f"  - Toplam metin: {method_stats['total_texts']}")
        print(f"  - Toplam eşleşme: {method_stats['total_matches']}")
        print(f"  - Ortalama güven: {method_stats['avg_confidence']:.1f}%")
        print(f"  - Ortalama benzerlik: {method_stats['avg_similarity']:.2f}")
    
    # En iyi yöntemi bul
    best_method = max(results_summary.keys(), key=lambda x: results_summary[x]['avg_similarity'])
    print(f"\n🏆 EN İYİ YÖNTEM: {best_method.upper()}")
    print(f"   Benzerlik: {results_summary[best_method]['avg_similarity']:.2f}")
    print(f"   Güven: {results_summary[best_method]['avg_confidence']:.1f}%")

if __name__ == "__main__":
    import cv2
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "detailed":
            detailed_test()
        else:
            print("Kullanım:")
            print("  python test_ocr.py          # Tek görüntü testi")
            print("  python test_ocr.py quick    # Hızlı test")
            print("  python test_ocr.py detailed # Detaylı test")
    else:
        test_single_image() 