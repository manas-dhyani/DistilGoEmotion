from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirement 
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
name ='DistilGoEmotion',
version ='0.0.1',
author='Manas Dhyani',
author_email='manasdhyani24@gmail.com',
description='Distilled GoEmotions model for emotion classification',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)