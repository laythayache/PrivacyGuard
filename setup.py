from setuptools import setup, find_packages

setup(
    name="privacyguard",
    version="0.1.0",
    description="Privacy-first edge AI de-identification library for real-time video.",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "onnxruntime",
        "numpy"
    ],
    python_requires=">=3.8",
)
