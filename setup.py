from setuptools import find_packages, setup


setup(
    name="xj0432-uav-emergency-response",
    version="1.0.0",
    description="YOLOv8 and deep reinforcement learning code for closed-loop UAV emergency response scheduling.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "pillow",
        "pyyaml",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
        "ultralytics",
        "torch",
        "tensorboard",
    ],
)
