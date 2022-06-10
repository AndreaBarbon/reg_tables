import setuptools

with open("README.md", "r") as fh: long_description = fh.read()

setuptools.setup(
    name="reg_tables",                     
    version="0.0.4",                     
    author="Andrea Barbon, Kirill Kazakov",                     
    author_email='andrea.3arbon@gmail.com',
    url = 'https://github.com/AndreaBarbon/reg_tables',
<<<<<<< Updated upstream
    download_url='https://github.com/AndreaBarbon/reg_tables/archive/refs/tags/0.0.2.tar.gz',
    description="",
=======
    description="A simple [linearmodels](https://pypi.org/project/linearmodels/) extension to run panel regressions with different specifications and export the results in a professional-looking latex table",
>>>>>>> Stashed changes
    license='MIT',
    long_description=long_description,      
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    
    keywords = ['econometrics', 'Linear models', 'regression table', 'panel regression', 'fixed effects', 'clustered standard errors'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                    
    python_requires='>=3.0',                
    py_modules=["reg_tables"],            
    install_requires=[
        'numpy',
        'pandas',
        'linearmodels'
    ]
)
