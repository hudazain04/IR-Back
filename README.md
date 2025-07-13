نظام استرجاع معلومات متكامل (Backend) مبني وفق بنية SOA لاسترجاع الوثائق باستخدام خوارزميات متعددة.

---

## 📁 هيكل المشروع

 IR-Project/
 │
 ├
 ├── notebooks/ # دفاتر Jupyter لتحليل النتائج والتقييم
 ├── data/ # بيانات SQLite + ملفات المعالجة 
 ├── models/ # نماذج قاعدة البيانات (SQLAlchemy)
 ├── services/ # الخدمات المسؤولة عن BM25، TF-IDF، Word2Vec، Hybrid
 ├── routes/ # نقاط النهاية (APIs)
 ├── repositories/ # التعامل مع قاعدة البيانات
 ├── database.py # الاتصال بقاعدة البيانات
 ├── main.py # نقطة البداية لتطبيق FastAPI
 ├── .gitignore # ملفات يجب تجاهلها في Git
 └── README.md # هذا الملف




---

## 🧠 الخوارزميات المدعومة

- Vector Space Model (TF-IDF)
- Word2Vec
- BM25
- Hybrid (دمج بين الخوارزميات السابقة)
- FAISS Index (اختياري لتحسين الأداء)

---

## 🚀 تشغيل المشروع

### ✅ Backend (FastAPI)
```bash
cd IR
python -m venv venv
source venv/bin/activate  # أو venv\\Scripts\\activate على ويندوز
pip install -r requirements.txt
uvicorn main:app --reload


---

🧪 التقييم (Evaluation)
تم حساب المقاييس التالية:

MAP

Precision@10

Recall@10

NDCG

