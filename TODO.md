## Steps to Update a Pip-Installable Package

1. **Update Version:**
   - Change the version in the `pyproject.toml` file.

2. **Build the Package:**
   - Clean the `dist` directory:
     ```bash
     rm -rf dist/*
     ```
   - Build the package:
     ```bash
     pip install build 
     python -m build
     python setup.py sdist bdist_wheel
     ```

3. **Get Token from PyPI:**
   - Obtain a new PyPI token for authentication.

4. **Upload to PyPI:**
   - Upload the package to PyPI using Twine:
     ```bash
     twine upload dist/* --verbose
     ```

5. **Verify Update Locally:**
   - Uninstall the old version:
     ```bash
     pip uninstall htree
     ```
   - Install the new version:
     ```bash
     pip install htree
     ```
   - Verify the installed version:
     ```bash
     pip show htree
     ```

This ensures that you properly update the package, upload it to PyPI, and verify the correct version is installed.
