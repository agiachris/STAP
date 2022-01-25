import setuptools

setuptools.setup(
    name="temporal-policies",
    version="0.0.1",
    author="Christopher Agoa",
    author_email="cagia@cs.stanford.edu",
    description="Learning compositional policies for long horizon planning.",
    url="https://github.com/agiachris/temporal-policies",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
)