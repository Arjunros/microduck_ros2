from setuptools import find_packages, setup

package_name = "microduck_sim"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arjun",
    maintainer_email="projecthumanoidx@gmail.com",
    description="MuJoCo simulation of Microduck for ROS 2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_node = microduck_sim.sim_node:main",
            "make_map = microduck_sim.mapgen:main",
            "make_urdf = microduck_sim.urdf:main",
            "camera_node = microduck_sim.camera_node:main",
        ],
    },
)
