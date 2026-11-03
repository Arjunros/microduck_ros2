from setuptools import find_packages, setup

package_name = "microduck_policy"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arjun",
    maintainer_email="projecthumanoidx@gmail.com",
    description="Microduck ONNX locomotion policy node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = microduck_policy.policy_node:main",
        ],
    },
)
