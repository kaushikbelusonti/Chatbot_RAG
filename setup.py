from setuptools import setup

setup(
    name="Chatbot_RAG",
    version="1.0",
    install_requires=[
        "package1>=1.0",
        "package2<2.0",
    ],
    python_requires=="==3.11.4",  # Enforce Python 3.11.4 specifically
)
