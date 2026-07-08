# استفاده از Python رسمی
FROM python:3.12-slim

# تعیین پوشه کاری
WORKDIR /app

RUN apt-get update && apt-get install -y git


# کپی فایل وابستگی‌ها
COPY requirements.txt .

# نصب کتابخانه‌ها
RUN pip install --no-cache-dir -r requirements.txt

# کپی کل پروژه
COPY . .

# اجرای پروژه
CMD ["python", "run_pipeline.py"]