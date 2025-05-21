from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mediapipe_hand_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.*")),
        )
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koener',
    maintainer_email='nic.koenig37@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_publisher = mediapipe_hand_publisher.hand_publisher:main'
        ],
    },
)
