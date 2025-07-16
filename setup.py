from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

setup(
    name="cv_arm",
    version="0.1.0",
    description="Webcam to MediaPipe Pose to DOFBOT angles",
    author="Nathan DiGilio, Nick Yaple",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'cv-pose-to-arm = cv_arm.pose_to_arm:app',
        ],
    },
)
