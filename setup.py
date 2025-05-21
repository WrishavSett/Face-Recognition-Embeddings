from setuptools import setup, find_packages

setup(
    name='face-recognition',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'insightface',
        'pinecone-client',
        'numpy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'realtime-face-search=app.main:main',
        ],
    },
    author='Wrishav',
    description='Real-time face recognition using Face Embeddings.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='face recognition real-time pinecone insightface opencv',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)