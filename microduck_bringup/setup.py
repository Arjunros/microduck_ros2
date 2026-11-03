from glob import glob

from setuptools import find_packages, setup

package_name = "microduck_bringup"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arjun",
    maintainer_email="projecthumanoidx@gmail.com",
    description="Launch and config for the Microduck ROS 2 simulation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "joy_skills = microduck_bringup.joy_skills:main",
        ],
    },
)
