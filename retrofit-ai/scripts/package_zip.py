import os
import zipfile

def create_zip_package(source_dir=r"C:\Users\rakes\.gemini\antigravity\scratch\retrofit-ai", output_zip=r"C:\Users\rakes\.gemini\antigravity\brain\6e27c7ae-0177-4dfd-83fa-e077d93d208d\retrofit-ai.zip"):
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    
    ignore_dirs = {'.git', '__pycache__', '.venv', '.pytest_cache'}
    ignore_extensions = {'.pyc', '.pyo'}
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in ignore_extensions:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
                
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[ZIP] Packaged repository into {output_zip} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    create_zip_package()
