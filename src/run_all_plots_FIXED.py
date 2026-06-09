"""
run_all_plots_FIXED.py

修正版正規化parquetを使って、全ての図を再生成する。
対象: DOY20-80 および DOY35-60 の主要解析スクリプト
"""
import subprocess
import sys
import os
from pathlib import Path

PYTHON = str(Path(__file__).parent.parent / "venv" / "bin" / "python")
SRC    = Path("src")

# matplotlib Aggバックエンドを使うため環境変数をセット
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# 実行対象スクリプト（順番に実行）
SCRIPTS = [
    # --- 正規化前後比較（2D DOY20-80） ---
    "plot_density_before_after_normalization_SWARM-A_DOY20-80_LT4-11_16-23.py",
    "plot_density_before_after_normalization_SWARM-B_DOY20-80_LT22-5_11-17_450km.py",
    "plot_density_before_after_normalization_SWARM-C_DOY20-80_LT4-11_16-23.py",

    # --- 2D 残差プロット (DOY20-80) ---
    "plot_2d_residual_SWARM-A_DOY20-80_LT4-11_16-23.py",
    "plot_2d_residual_SWARM-B_DOY20-80_LT22-5_11-17_450km.py",
    "plot_2d_residual_SWARM-C_DOY20-80_LT4-11_16-23.py",

    # --- 2D 残差プロット (DOY35-60) ---
    "plot_2d_residual_SWARM-A_DOY35-60_LT4-11_16-23.py",
    "plot_2d_residual_SWARM-B_DOY35-60_LT22-5_11-17_450km.py",
    "plot_2d_residual_SWARM-C_DOY35-60_LT4-11_16-23.py",

    # --- 1D 残差プロット (DOY35-60) ---
    "1D_residual_density_SWARM-A_DOY35-60_LT4-11_16-23.py",
    "1D_residual_density_SWARM-B_DOY35-60_LT22-5_11-17_450km.py",
    "1D_residual_density_SWARM-C_DOY35-60_LT4-11_16-23.py",

    # --- 1D 残差プロット (DOY20-80) ---
    "1D_residual_density_SWARM-B_DOY20-80_LT22-5_11-17_450km.py",
    "1D_residual_density_SWARM-C_DOY20-80_LT4-11_16-23.py",
]

ok_list   = []
fail_list = []

for script in SCRIPTS:
    script_path = SRC / script
    if not script_path.exists():
        print(f"[SKIP]  {script}  ← ファイルなし")
        continue

    print(f"\n[RUN]   {script}")
    result = subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=True, text=True, env=env
    )
    if result.returncode == 0:
        print(f"  ✅ 成功")
        # 最終出力行だけ表示
        last = [l for l in result.stdout.splitlines() if l.strip()]
        if last:
            print(f"  {last[-1]}")
        ok_list.append(script)
    else:
        print(f"  ❌ 失敗 (exit={result.returncode})")
        # エラー末尾5行を表示
        err_lines = result.stderr.splitlines() + result.stdout.splitlines()
        for l in err_lines[-8:]:
            print(f"    {l}")
        fail_list.append(script)

print("\n" + "="*60)
print(f"✅ 成功: {len(ok_list)}/{len(SCRIPTS)}")
if fail_list:
    print(f"❌ 失敗: {len(fail_list)}")
    for s in fail_list:
        print(f"  - {s}")
print("="*60)
