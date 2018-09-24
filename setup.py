from setuptools import setup

setup(name='quanser_robots',
      version='0.0.3',
      description='Simulated environments and Quanser robot control interfaces',
      url='https://git.ias.informatik.tu-darmstadt.de/quanser/clients',
      author='Intelligent Autonomous Systems Lab',
      author_email='boris@robot-learning.de',
      license='MIT',
      packages=['quanser_robots', 'quanser_robots.qube'],
      zip_safe=False,
      install_requires=[
          'gym>=0.10.5', 'numpy>=1.15.1', 'scipy>=1.1.0', 'matplotlib>=2.2.3',
          'vpython>=7.4.6'
      ])
