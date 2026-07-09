"""
check_gfz_data.py
Download Kp_ap_Ap_SN_F107_since_1932.txt from GFZ Potsdam and inspect the header structure.
"""
import urllib.request

url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
print(f"Downloading from {url}...")

# 最初の100行だけ読み込んで確認
lines = []
with urllib.request.urlopen(url) as response:
    for _ in range(150):
        line = response.readline().decode('utf-8')
        if not line:
            break
        lines.append(line)

print("\n=== Header (first 100 lines) ===")
for i, line in enumerate(lines[:100]):
    print(f"{i+1:3d}: {line.strip()}")

print("\n=== Data preview (lines 100-150) ===")
for i, line in enumerate(lines[100:150]):
    print(f"{i+101:3d}: {line.strip()}")
