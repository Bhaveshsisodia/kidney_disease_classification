import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version = "0.0.0"

REPO_NAME = "kidney_disease_classification"
AUTHOR_USER_NAME = "Bhavesh Kumar Lohar"
SRC_REPO = "kidney_disease_classifier"
AUTHOR_EMAIL ="bhaveshsisodia2@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A machine learning project for classifying kidney disease.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",

    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"))