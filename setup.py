from setuptools import setup

install_requires = [
    'tensorflow-gpu==1.2.1',
    'Keras==2.0.8',
    'numpy==1.13.3',
    'Pillow==4.3.0',
    'pandas==0.20.3'
]

setup(
    name='kerasdcgan',
    description='Deep Convolutional Generative Adversarial Network in Keras',
    version='0.1',
    packages=['kerasdcgan'],
    install_requires=install_requires
)