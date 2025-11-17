# ASR Model Comparison Report
**Generated:** 2025-11-09T15:30:57.741899
**Number of models:** 3
**Number of samples:** 765

---

## Overall Metrics

| Model | WER | CER | MER | Avg RTF | Samples |
|-------|-----|-----|-----|---------|----------|
| whisper-large-v3-turbo | 0.1112 | 0.0306 | 0.1098 | 0.113 | 765 |
| final_model.nemo | 0.1744 | 0.0426 | 0.1713 | 0.030 | 765 |
| malaysian-whisper-large-v3-turbo-v3 | 0.3032 | 0.1427 | 0.2496 | 0.139 | 765 |

### 🏆 Best Model
**whisper-large-v3-turbo** - WER: 0.1112

### ⚠️ Worst Model
**malaysian-whisper-large-v3-turbo-v3** - WER: 0.3032

---

## Top 20 Samples with Highest Disagreement

These are samples where models produce significantly different results:

### 1. 633.wav

**Reference:** `26 4`

**WER Variance:** 6000.0% (range: 100.0% - 6100.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=100.0%): `26.4`
- **final_model.nemo** (WER=100.0%): `264`
- **malaysian-whisper-large-v3-turbo-v3** (WER=6100.0%): `26 4. 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27 28 29 30 31 32 33 34 35 36 37 38 39 40 42 45 46 47 48 49 50 51 52 53 54 56 57 58 59 60 62 63 67 68 65 55 66 69 61 72 73 43 77 82 92 93 94 96 97 87 98 90 91 95 99 80 70 tahun 2000 hari ini bergaduh ke belaknya dikali ni yang terpikani dia kata kali gaya nilai kanan itu pergi dahi bahagihkan saya tidak memandanginya ia adalah perkara lain dalam hal ehmah mereka mengatakan anda boleh menjadi lebih baik-tidahi orang tua pada masa lagi?`

---

### 2. 441.wav

**Reference:** `pemerhatian jarak jauh melalui`

**WER Variance:** 3500.0% (range: 25.0% - 3525.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=25.0%): `Pemerhatian jarak jauh melalui`
- **final_model.nemo** (WER=25.0%): `Pemerhatian jarak jauh melalui`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3525.0%): `Pemerhatian jarak jauh melalui. 3, 2, 1, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 49,, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 65, 67, 61, 72, 73, 64, 68, 69, 71, 79, 77, 78, 81, 82, 93, 91, 94, 97, 87, 88, 98, 92, 95, 89, 90, 80, 99, 900, 96, 85, 750, 800, 720, 700, 197, 70, 103, 400, 300, 450, 550, 500, 1500, 150,900, 170, 130, 240, 330, 160, 127,700, 250, 1400, 320, 140, 120, 180, 230, 220, 650, 350,150, 200,5000, 270, 1900, 1200, 280, 1300, 128,800, Jia, 5000, 6000, 1000, 3000, 4000, 600,hyun, 100,750,350, 911,`

---

### 3. 296.wav

**Reference:** `kripton pada tahun biro`

**WER Variance:** 3325.0% (range: 50.0% - 3375.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=50.0%): `Krypton pada tahun biru`
- **final_model.nemo** (WER=50.0%): `Crifton pada tahun Biro`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3375.0%): `Kripton pada tahun biru. 3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 49 , 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 65, 67, 61, 72, 73, 64, 68, 69, 71, 79, 77, 82, 93, 91, 92, 94, 97, 81, 98, 95, 90, 99, 80, 900, 96, 87, 89, 88, 85, 750, 720, 800, 102, 450, 330, 320, 240, 550, 103, 400, 130, 220, 230, 250, 650, 170, 150, 160, 127, 300, 120, 700, 180, 270, 350, 1500, 280, 140, 500,900, 200, 128,700,150, 1200, 1400, 1900, Jia, 600, 1300, 197,hana, 1000,5000, 101,800,hyun,`

---

### 4. 97.wav

**Reference:** `bumi bulan selari gerhana`

**WER Variance:** 3250.0% (range: 25.0% - 3275.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=25.0%): `Bumi bulan selari gerhana`
- **final_model.nemo** (WER=25.0%): `Bumi bulan selari gerhana`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3275.0%): `Bumi bulan selari gerhana. 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 49 , 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 65, 67, 61, 72, 73, 71, 82, 93, 91, 92, 94, 97, 79, 81, 98, 95, 90, 99, 80, 96, 87, 89, 88, 85, 900, 800, 750, 77, 83, 700, 720, 240, 450, 320, 400, 500, 300, 330, 550, 250, 1500, 150, 160, 170, 130, 230, 220, 650, 127, 270, 120, 350, 280, 140, 180, 200, 1400, 1200, 128,900, 1900, 102,150,700, 1300,hyun, Jia, 1000, Palestin, Zhan, unforgettable,><`

---

### 5. 154.wav

**Reference:** `selepas lebih kurang tiga`

**WER Variance:** 3200.0% (range: 25.0% - 3225.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `Selepas lebih kurang tiga`
- **whisper-large-v3-turbo** (WER=50.0%): `Selepas lebih kurang 3.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3225.0%): `Selepas lebih kurang tiga. 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 49 , 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 65, 67, 61, 72, 73, 71, 82, 93, 91, 92, 94, 97, 79, 81, 83, 98, 87, 88, 89, 90, 80, 99, 95, 85, 96, 77, 900, 800, 750, 720, 240, 320, 330, 450, 550,900, 220, 230, 280, 250, 650, 350, 400, 300, 500, 1500, 150, 170, 130, 160, 127,700, 270, 700, 140,150,5000, 1400, 200, 1900, 1200, 120,800, 180, 1300, 197,hyun, Jia, 128,350,130, airlift,hana, Palestin, unforgettable,><`

---

### 6. 638.wav

**Reference:** `pejabat taman negara kuala`

**WER Variance:** 3175.0% (range: 0.0% - 3175.0%)

**Hypotheses:**
- **final_model.nemo** (WER=0.0%): `pejabat taman negara kuala`
- **whisper-large-v3-turbo** (WER=100.0%): `Pejabat Taman Negara Kuala`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3175.0%): `Pejabat Taman Negara Kuala. 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 59, 51, 52, 53, 54, 55, 56, 57, 58, 67, 63, 62, 65, 61, 72, 73, 64, 69, 71, 79, 77, 81, 82, 93, 91, 92, 94, 97, 83, 98, 90, 95, 99, 80, 900, 96, 87, 89, 88, 85, 800, 750, 720,900, 103, 911, 102, 450, 550, 330, 240, 320, 400, 130, 500, 150, 160, 170, 300, 140, 1500, 120, 250, 700, 230, 180, 220, 127,700, 650, 1400, 350,150,5000, 1200,800,130, 1900, 1300, 128,750,200,500, 1000,400,300,350,100,1000,600,hyun,`

---

### 7. 377.wav

**Reference:** `ramai pendakian yang bermula`

**WER Variance:** 3150.0% (range: 0.0% - 3150.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=0.0%): `ramai pendakian yang bermula`
- **final_model.nemo** (WER=25.0%): `Ramai pendakian yang bermula`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3150.0%): `Ramai pendakian yang bermula. 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 49 , 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 65, 67, 61, 72, 73, 71, 82, 77, 81, 83, 93, 79, 91, 92, 94, 97, 87, 98, 89, 90, 95, 99, 80, 900, 96, 88, 85, 750, 800, 720, 320, 240, 450, 550, 330, 400, 500, 650, 300, 250, 150,900, 160, 170, 130, 1500, 230, 220, 350, 127,700, 140, 1400, 120, 700, 180,150, 1200, 200, 1900, 128,5000, 270, 102,800,hyun, Jia, Zhan,`

---

### 8. 312.wav

**Reference:** `ditemukan beberapa bahagian dunia`

**WER Variance:** 3050.0% (range: 25.0% - 3075.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `Ditemukan beberapa bahagian dunia`
- **whisper-large-v3-turbo** (WER=50.0%): `Ditemukan beberapa bahagian dunia.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=3075.0%): `Ditemukan beberapa 2 bahagian dunia. 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 61, 67, 68, 69, 73, 72, 65, 71, 77, 79, 82, 81, 83, 93, 94, 85, 87, 88, 89, 91, 92, 98, 97, 99, 90, 95, 80, 96, 750, 900, 800, 103, 197, 102, 320, 330, 240, 720,900, 1080p 120, 130, 230, 220, 450, 550, 170, 280, 150, 160, 140, 1500, 250, 180, 300, 400, 1400, 500,150, 650, 1900, 127,700, 1300, 1200`

---

### 9. 381.wav

**Reference:** `saiz kaki diameter inci`

**WER Variance:** 175.0% (range: 25.0% - 200.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `size kaki diameter inci`
- **whisper-large-v3-turbo** (WER=50.0%): `Saiz kaki diameter inci.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=200.0%): `Saiz kaki diameter inci. 2, 3, 4 , 5 .`

---

### 10. 116.wav

**Reference:** `tumbuh tumbuhan lain rafflesia`

**WER Variance:** 125.0% (range: 75.0% - 200.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=75.0%): `Tumbuhan-tumbuhan lain reflisya.`
- **final_model.nemo** (WER=75.0%): `Tumbuhan-tumbuhan lain reflexia`
- **malaysian-whisper-large-v3-turbo-v3** (WER=200.0%): `Tumbuhan-tumbuhan lain, Reflecia. Hai, hai, hai! Hai, hi!`

---

### 11. 390.wav

**Reference:** `sebut perkataan tersenarai`

**WER Variance:** 100.0% (range: 33.3% - 133.3%)

**Hypotheses:**
- **final_model.nemo** (WER=33.3%): `Sebut perkataan tersenarai`
- **whisper-large-v3-turbo** (WER=133.3%): `Sebut perkataan, terus cenarai.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=133.3%): `Sebut perkataan. Terus senarai.`

---

### 12. 537.wav

**Reference:** `tolong sebut biperforate`

**WER Variance:** 100.0% (range: 33.3% - 133.3%)

**Hypotheses:**
- **final_model.nemo** (WER=33.3%): `Tolong sebut biperforate`
- **whisper-large-v3-turbo** (WER=133.3%): `Tolong sabut by perforate.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=133.3%): `Tolong sebut by perforate. Tidak`

---

### 13. 95.wav

**Reference:** `the fight`

**WER Variance:** 100.0% (range: 0.0% - 100.0%)

**Hypotheses:**
- **final_model.nemo** (WER=0.0%): `the fight`
- **whisper-large-v3-turbo** (WER=100.0%): `DeFi.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=100.0%): `The Fight`

---

### 14. 507.wav

**Reference:** `sepanyol juta pada tahun`

**WER Variance:** 100.0% (range: 25.0% - 125.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `Sepanjol juta pada tahun`
- **whisper-large-v3-turbo** (WER=50.0%): `Sepanyol juta pada tahun.`
- **malaysian-whisper-large-v3-turbo-v3** (WER=125.0%): `Sepanyol juta pada tahun? 10.001,000 ,0192,0203,0216,0254,0275,0337,0358,0459,0479, 04410,049,055,0659,069,089,116,100,075,077,089,099,095,097,090,093,098,091,096,087,094,088,080,0900,092,085,0999,086,084,083,081,028,048,040,050,060,070,038,039,043,046,041,064,058,068,067,063,079,078,0800,082,018,0180,030,036,0400,034,037,0150,042,0600,032,0500,0300,0200,0700,0250,024,026,029,023,0++,0350,056,031,05000,066,022,017,016,051,057,053,054,052,0101,059,061,01000,069,071,015,0120,0PM,02021,0umn,0hma,`

---

### 15. 566.wav

**Reference:** `terbaharu pelancongan mesir`

**WER Variance:** 100.0% (range: 0.0% - 100.0%)

**Hypotheses:**
- **final_model.nemo** (WER=0.0%): `terbaharu pelancongan mesir`
- **whisper-large-v3-turbo** (WER=100.0%): `Terbaharu Pelancongan Mesir`
- **malaysian-whisper-large-v3-turbo-v3** (WER=100.0%): `Terbaharu Pelancongan Mesir.`

---

### 16. 684.wav

**Reference:** `ini`

**WER Variance:** 100.0% (range: 0.0% - 100.0%)

**Hypotheses:**
- **whisper-large-v3-turbo** (WER=0.0%): `ini`
- **malaysian-whisper-large-v3-turbo-v3** (WER=0.0%): `ini`
- **final_model.nemo** (WER=100.0%): `Ini`

---

### 17. 744.wav

**Reference:** `daratan ini berbeza dengan`

**WER Variance:** 100.0% (range: 0.0% - 100.0%)

**Hypotheses:**
- **malaysian-whisper-large-v3-turbo-v3** (WER=0.0%): `daratan ini berbeza dengan`
- **whisper-large-v3-turbo** (WER=25.0%): `Daratan ini berbeza dengan`
- **final_model.nemo** (WER=100.0%): `kita ni Dia datang ini berbeza dengan`

---

### 18. 54.wav

**Reference:** `sayangnya rasulullah pada kita`

**WER Variance:** 75.0% (range: 25.0% - 100.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `sayangnya rasullah pada kita`
- **malaysian-whisper-large-v3-turbo-v3** (WER=75.0%): `Sayangnya Rasulullah pada kita.`
- **whisper-large-v3-turbo** (WER=100.0%): `Sayangnya Rasulullah kepada kita.`

---

### 19. 176.wav

**Reference:** `hentaman komet ketiadaan udara`

**WER Variance:** 75.0% (range: 25.0% - 100.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `Rentaman komet ketiadaan udara`
- **malaysian-whisper-large-v3-turbo-v3** (WER=75.0%): `Hentaman Komet ketiadaan udara.`
- **whisper-large-v3-turbo** (WER=100.0%): `Hentaman Komet Ketiadaan Udara`

---

### 20. 234.wav

**Reference:** `ahli geologi sedar akan`

**WER Variance:** 75.0% (range: 25.0% - 100.0%)

**Hypotheses:**
- **final_model.nemo** (WER=25.0%): `Ahli geologi sedar akan`
- **malaysian-whisper-large-v3-turbo-v3** (WER=50.0%): `Ahli geologi sedar akan.`
- **whisper-large-v3-turbo** (WER=100.0%): `Ahli Geologi sedarakan`

---
