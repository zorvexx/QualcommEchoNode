import os

def find_file(filename, search_path):
    matches = []
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            matches.append(os.path.join(root, filename))
    return matches

print("Searching for unified_dataset_14col.csv and mlx90614_dataset_converted.csv...")
m1 = find_file("unified_dataset_14col.csv", r"C:\Users\rakes")
m2 = find_file("mlx90614_dataset_converted.csv", r"C:\Users\rakes")

print("Found unified_dataset_14col.csv:", m1)
print("Found mlx90614_dataset_converted.csv:", m2)
