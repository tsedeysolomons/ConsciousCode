#!/usr/bin/env python3
"""
ConsciousCode - Environment Setup Verification
"""

import sys
import subprocess
import os

def check_package(package_name, import_name=None):
    """Check if a package is installed and get version"""
    if import_name is None:
        import_name = package_name
        
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"{package_name}: {version}")
        return True
    except ImportError:
        print(f"{package_name}: NOT INSTALLED")
        return False

def main():
    print("CONSCIOUSCODE - ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Python 3.8 or higher is required!")
        return False
    
    # Check essential packages
    print("\nCHECKING ESSENTIAL PACKAGES:")
    packages = [
        ('NumPy', 'numpy'),
        ('Matplotlib', 'matplotlib'), 
        ('Jupyter', 'jupyter')
    ]
    
    all_ok = True
    for name, import_name in packages:
        if not check_package(name, import_name):
            all_ok = False
    
    # Create directory structure
    print("\nCREATING PROJECT STRUCTURE:")
    directories = ['experiments', 'data', 'research', 'visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}/")
        else:
            print(f"Exists: {directory}/")
    
    # Final verification
    print("\n" + "=" * 50)
    if all_ok:
        print("CONSCIOUSCODE ENVIRONMENT READY!")
        print("   You can now begin consciousness experiments!")
        print("\nNext: Run 'python test_numpy_basics.py'")
    else:
        print("Some packages missing. Run: pip install numpy matplotlib jupyter")
    
    return all_ok

if __name__ == "__main__":
    main()