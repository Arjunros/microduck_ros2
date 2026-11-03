from setuptools import find_packages, setup

package_name = "microduck_agent"

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
    description="Natural-language and voice control for Microduck",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "agent_node = microduck_agent.agent_node:main",
            "chat = microduck_agent.chat:main",
            "voice = microduck_agent.voice:main",
        ],
    },
)
