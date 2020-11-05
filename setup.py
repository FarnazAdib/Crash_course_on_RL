import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Reinforcement_Learning_for_Control",
    version="0.0.1",
    author="Farnaz Adib Yaghmaie",
    author_email="farnaz.adib.yaghmaei@gmail.com",
    description="Basic RL routines for control problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FarnazAdib/Crash_course_on_RL.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LiU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)