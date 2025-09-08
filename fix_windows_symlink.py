#!/usr/bin/env python3
# fix_windows_symlink.py - Fix for Windows symlink permission issues

import os
import sys
import shutil
from pathlib import Path

def clean_model_directories():
    """Clean up versioned directories that cause symlink issues on Windows"""
    
    print("🧹 Cleaning up model directories to fix Windows symlink issues...")
    print("=" * 60)
    
    # Find and clean versioned directories
    models_dir = Path("models/saved_models")
    
    if models_dir.exists():
        # Look for Patchcore directories
        patchcore_dirs = list(models_dir.glob("**/Patchcore"))
        
        for patchcore_dir in patchcore_dirs:
            print(f"📁 Found: {patchcore_dir}")
            
            # Look for versioned directories (v1, v2, etc.)
            versioned_dirs = list(patchcore_dir.glob("**/v*"))
            
            for ver_dir in versioned_dirs:
                if ver_dir.name.startswith("v") and ver_dir.name[1:].isdigit():
                    print(f"   🗑️ Removing versioned directory: {ver_dir}")
                    try:
                        shutil.rmtree(ver_dir)
                    except Exception as e:
                        print(f"   ⚠️ Could not remove {ver_dir}: {e}")
            
            # Remove 'latest' symlinks
            latest_links = list(patchcore_dir.glob("**/latest"))
            for latest in latest_links:
                if latest.is_symlink() or latest.exists():
                    print(f"   🗑️ Removing 'latest' link/directory: {latest}")
                    try:
                        if latest.is_dir() and not latest.is_symlink():
                            shutil.rmtree(latest)
                        else:
                            latest.unlink()
                    except Exception as e:
                        print(f"   ⚠️ Could not remove {latest}: {e}")
    
    print()
    print("✅ Cleanup completed!")
    print()
    print("📝 Next steps:")
    print("1. The training script has been updated to disable versioned directories")
    print("2. Run training again: python train.py")
    print("3. Models will now save directly without version numbers")

def check_admin_rights():
    """Check if script is running with admin rights on Windows"""
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        return is_admin != 0
    except:
        return False

def main():
    """Main function"""
    
    print("🔧 Windows Symlink Fix for PatchCore Training")
    print("=" * 60)
    
    # Check OS
    if os.name != 'nt':
        print("ℹ️ This script is for Windows only.")
        print("   Your OS doesn't have symlink permission issues.")
        return
    
    # Check admin rights
    if check_admin_rights():
        print("✅ Running with administrator privileges")
    else:
        print("⚠️ Not running as administrator")
        print("   Symlink creation may fail on Windows without admin rights")
    
    print()
    
    # Clean up directories
    clean_model_directories()
    
    print("=" * 60)
    print("🎯 Solutions implemented:")
    print()
    print("1. ✅ Updated train_model.py to disable versioned directories")
    print("2. ✅ Cleaned up existing versioned directories")
    print()
    print("Alternative solutions if issues persist:")
    print()
    print("Option A: Run PowerShell as Administrator:")
    print("   1. Right-click PowerShell -> 'Run as Administrator'")
    print("   2. Navigate to project: cd F:\\Rpi_CAM\\dev\\patchcore_defect_detection")
    print("   3. Run: python train.py")
    print()
    print("Option B: Enable Developer Mode (Windows 10/11):")
    print("   1. Settings -> Update & Security -> For Developers")
    print("   2. Enable 'Developer Mode'")
    print("   3. This allows symlinks without admin rights")
    print()
    print("Option C: Use WSL2 (Windows Subsystem for Linux):")
    print("   Symlinks work without issues in WSL2")

if __name__ == "__main__":
    main()