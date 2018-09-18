from setuptools import setup

setup(name='quanser_clients',
      version='0.0.1',
      description='Simulated and real robot environments for Quanser robots',
      url='https://git.ias.informatik.tu-darmstadt.de/quanser/clients',
      author='Intelligent Autonomous Systems Lab',
      author_email='boris@robot-learning.de',
      license='MIT',
      packages=['qube'],
      zip_safe=False,
      install_requires=[
          'gym>=0.10.5', 'numpy>=1.15.1', 'scipy>=1.1.0', 'matplotlib>=2.2.3',
          'vpython>=7.4.6'
      ])
