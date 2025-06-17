from setuptools import setup, find_packages

setup(
    name='face-recognition',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'insightface',
        'pinecone-client',   # If you're using Pinecone anywhere
        'numpy',
        'matplotlib',
        'onnxruntime-silicon',  # Optimized for Apple Silicon; replace for other platforms if needed
        'elevenlabs',
        'playsound',
    ],
    entry_points={
        'console_scripts': [
            'realtime-face-search=localStore_app:main',
        ],
    },
    author='Wrishav',
    description='Real-time face recognition using Face Embeddings with local FAISS support and audio feedback.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='face recognition real-time faiss insightface opencv elevenlabs',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
