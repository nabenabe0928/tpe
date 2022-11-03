import os


start_dir = "results/"

for t in os.walk(start_dir):
    for f in t[-1]:
        file_path = os.path.join(t[0], f)
        file_size = os.path.getsize(file_path)
        if file_size < 10:
            print(f"Remove {file_path}")
            os.remove(file_path)
