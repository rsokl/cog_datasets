from setuptools import setup, find_packages


def do_setup():
    setup(name='Datasets',
          version="0.0",
          author='Ryan Soklaski',
          description='Collection of data sets for machine learning (CogWorks)',
          license='MIT',
          platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
          packages=find_packages())

if __name__ == "__main__":
    do_setup()
